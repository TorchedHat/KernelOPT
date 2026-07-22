# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5060 ns — **Best latency**: 4900 ns (**1.0327×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`replace_division_with_rsqrt_to_enable_fma_fusion`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_24_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3920,1,1), Waves/SM: 1.86
- **Registers/thread**: 18
- **Duration**: 5060 ns
- **SM Throughput**: 33.33 % | **Memory**: 12.24 % | **DRAM**: 8.13 %
- **Achieved Occupancy**: 62.13 %
- **IPC Active**: 2.43 inst/cycle | **No Eligible**: 41.43 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_to_eliminate_partial_wave | 0.982× | 0.958× | ✓ |  |
| B | autotune_xblock_eliminate_partial_wave | 1.001× | 0.846× | ✓ | **✓ best** |
| C | vectorized_loads_stores_via_larger_xblock | 0.973× | 0.322× | ✓ |  |
| D | increase_xblock_to_eliminate_partial_wave | 0.973× | 0.864× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | autotune_xblock_e... | Δ | vectorized_loads_... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5060 ns | 5280 ns | +220 ns (+4.3%) | 5980 ns | +920 ns (+18.2%) | 15740 ns | +10680 ns (+211.1%) | 5860 ns | +800 ns (+15.8%) |
| SM Throughput (%) | 33.33 % | 41.13 % | +7.80 pp | 52.55 % | +19.22 pp | 75.66 % | +42.33 pp | 53.87 % | +20.54 pp |
| Memory Throughput (%) | 12.24 % | 14.51 % | +2.27 pp | 14.18 % | +1.94 pp | 14.64 % | +2.40 pp | 14.51 % | +2.27 pp |
| DRAM Throughput (%) | 8.13 % | 7.79 % | -0.34 pp | 6.93 % | -1.20 pp | 2.61 % | -5.52 pp | 7.04 % | -1.09 pp |
| Achieved Occupancy (%) | 62.13 % | 68.78 % | +6.65 pp | 74.48 % | +12.35 pp | 84.84 % | +22.71 pp | 73.85 % | +11.72 pp |
| Registers/Thread | 18.00 register/thread | 24.00 register/thread | +6 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 1.86 | 1.86 | +0.00 | 1.86 | +0.00 | 7.42 | +5.56 | 1.86 | +0.00 |
| No Eligible (%) | 41.43 % | 21.73 % | -19.70 pp | 18.24 % | -23.19 pp | 12.28 % | -29.15 pp | 18.22 % | -23.21 pp |
| IPC Active | 2.43 inst/cycle | 3.06 inst/cycle | +0.63 | 3.17 inst/cycle | +0.74 | 3.45 inst/cycle | +1.02 | 3.23 inst/cycle | +0.80 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_eliminate_partial_wave` (0.846×)

Adding @triton.autotune with larger XBLOCK configs (512, 1024, 2048) to a pointwise kernel caused a regression (0.85x) despite improving scheduler eligibility (No Eligible: 41% → 18%) and occupancy (62% → 74%). The autotuner selected a config that nearly doubled register usage per thread (18 → 32), which increased total executed instructions by ~86% (1.35M → 2.51M) and extended kernel duration from 5.06 µs to 5.98 µs. The grid was not updated to use a lambda (META['XBLOCK']) — it remained hardcoded at cdiv(xnumel, 128) = 3920 blocks — so the larger XBLOCK configs ran with the same grid size, causing each thread block to process more elements than intended and inflating instruction count without reducing block count. The structural anti-pattern is adding @triton.autotune without also changing the grid launch to use `lambda META: (triton.cdiv(xnumel, META['XBLOCK']),)`, which means the grid size does not shrink when XBLOCK grows.

> **Applicable when**: Pointwise kernels with a partial-wave tail (LaunchConfiguration NCU rule), low register usage (<24 registers/thread), no shared memory, and a fixed small XBLOCK (e.g. 128) that produces many small blocks.

> **Avoid when**: The grid launch is not updated to use a META lambda alongside the autotune decorator — this causes the grid to remain at the original small-XBLOCK size while the kernel body processes more elements, inflating instruction count and causing regression. Also avoid when register count is already near the per-SM limit, as larger XBLOCK configs may push register usage up and reduce occupancy.

#### ❌ Plan D: `increase_xblock_to_eliminate_partial_wave` (0.864×)

Adding @triton.autotune to explore larger XBLOCK values (512, 1024, 2048) to reduce grid size and eliminate partial-wave tail caused a regression. Although occupancy improved (62% → 74%) and scheduler idle slots dropped (41% → 18%), registers per thread jumped from 18 to 32, nearly doubling total executed instructions (1.35M → 2.51M) and increasing elapsed cycles by 15%. The autotuner selected a config that increased register pressure and instruction count enough to outweigh the partial-wave elimination benefit, resulting in a net 13.6% slowdown. When the kernel has very low register usage (≤18/thread) and the partial-wave tail is the flagged bottleneck, simply adding autotune without controlling register pressure can cause the compiler to spill or expand register file usage at larger tile sizes, negating the wave-elimination gain.

> **Applicable when**: Kernel has LaunchConfiguration rule triggered with large partial-wave tail AND registers per thread remain stable (do not increase) at larger XBLOCK values

> **Avoid when**: Autotune causes registers per thread to increase significantly (e.g., 18 → 32); when the grid lambda is not updated to use meta['XBLOCK'] so the grid size does not actually shrink; when instruction count more than doubles at larger tile sizes

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_to_enable_partial_wave_elimination | 0.983× | 0.953× | ✓ |  |
| B | enable_fma_fusion_via_rsqrt_and_fix_grid_lambda | 1.026× | 1.033× | ✓ | **✓ best** |
| C | fix_grid_lambda_and_enable_fma_fusion | 0.992× | 0.878× | ✓ |  |
| D | reduce_nonfused_fp32_instructions_via_arithmetic_simplification | 0.972× | 0.325× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | enable_fma_fusion... | Δ | fix_grid_lambda_a... | Δ | reduce_nonfused_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5060 ns | 5310 ns | +250 ns (+4.9%) | 4900 ns | -160 ns (-3.2%) | 5760 ns | +700 ns (+13.8%) | 15580 ns | +10520 ns (+207.9%) |
| SM Throughput (%) | 33.33 % | 40.94 % | +7.61 pp | 30.09 % | -3.24 pp | 54.21 % | +20.88 pp | 75.62 % | +42.29 pp |
| Memory Throughput (%) | 12.24 % | 14.38 % | +2.14 pp | 19.08 % | +6.84 pp | 14.65 % | +2.41 pp | 14.62 % | +2.38 pp |
| DRAM Throughput (%) | 8.13 % | 7.77 % | -0.36 pp | 8.43 % | +0.30 pp | 7.15 % | -0.98 pp | 2.64 % | -5.49 pp |
| Achieved Occupancy (%) | 62.13 % | 69.74 % | +7.61 pp | 56.53 % | -5.60 pp | 73.66 % | +11.53 pp | 84.87 % | +22.74 pp |
| Registers/Thread | 18.00 register/thread | 24.00 register/thread | +6 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 1.86 | 1.86 | +0.00 | 1.86 | +0.00 | 1.86 | +0.00 | 7.42 | +5.56 |
| No Eligible (%) | 41.43 % | 24.06 % | -17.37 pp | 31.15 % | -10.28 pp | 22.36 % | -19.07 pp | 12.31 % | -29.12 pp |
| IPC Active | 2.43 inst/cycle | 3.08 inst/cycle | +0.65 | 2.72 inst/cycle | +0.29 | 3.23 inst/cycle | +0.80 | 3.44 inst/cycle | +1.01 |

---

## Iteration 3

**Best so far**: 1.0327× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_static_grid_bug | 1.050× | 1.026× | ✓ | **✓ best** |
| B | fix_static_grid_to_meta_lambda_for_autotune | 0.985× | 0.953× | ✓ |  |
| C | fix_static_grid_lambda_to_activate_autotune | 0.993× | 0.864× | ✓ |  |
| D | fix_static_grid_to_meta_lambda_to_eliminate_overlaunch | 0.972× | 0.321× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_static_grid_t... | Δ | fix_static_grid_l... | Δ | fix_static_grid_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5060 ns | 4930 ns | -130 ns (-2.6%) | 5310 ns | +250 ns (+4.9%) | 5860 ns | +800 ns (+15.8%) | 15780 ns | +10720 ns (+211.9%) |
| SM Throughput (%) | 33.33 % | 30.24 % | -3.09 pp | 41.20 % | +7.87 pp | 54.00 % | +20.67 pp | 75.04 % | +41.71 pp |
| Memory Throughput (%) | 12.24 % | 18.97 % | +6.73 pp | 14.61 % | +2.37 pp | 14.61 % | +2.37 pp | 14.53 % | +2.29 pp |
| DRAM Throughput (%) | 8.13 % | 8.41 % | +0.28 pp | 7.76 % | -0.37 pp | 7.05 % | -1.08 pp | 2.61 % | -5.52 pp |
| Achieved Occupancy (%) | 62.13 % | 57.13 % | -5.00 pp | 69.21 % | +7.08 pp | 73.80 % | +11.67 pp | 84.60 % | +22.47 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 24.00 register/thread | +6 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 1.86 | 1.86 | +0.00 | 1.86 | +0.00 | 1.86 | +0.00 | 7.42 | +5.56 |
| No Eligible (%) | 41.43 % | 31.57 % | -9.86 pp | 22.57 % | -18.86 pp | 18.59 % | -22.84 pp | 12.33 % | -29.10 pp |
| IPC Active | 2.43 inst/cycle | 2.73 inst/cycle | +0.30 | 3.01 inst/cycle | +0.58 | 3.24 inst/cycle | +0.81 | 3.50 inst/cycle | +1.07 |

---

## Iteration 4

**Best so far**: 1.0327× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_to_meta_lambda_for_existing_autotune | 1.038× | 0.994× | ✓ |  |
| B | fix_grid_lambda_and_simplify_fp32_arithmetic_for_fma | 0.985× | 0.953× | ✓ |  |
| C | fix_grid_lambda_to_correctly_activate_autotune | 0.992× | 0.841× | ✓ |  |
| D | fix_static_grid_lambda_to_match_autotuned_xblock | 0.993× | 0.327× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_t... | Δ | fix_grid_lambda_a... | Δ | fix_grid_lambda_t... | Δ | fix_static_grid_l... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5060 ns | 5090 ns | +30 ns (+0.6%) | 5310 ns | +250 ns (+4.9%) | 6020 ns | +960 ns (+19.0%) | 15490 ns | +10430 ns (+206.1%) |
| SM Throughput (%) | 33.33 % | 29.43 % | -3.90 pp | 41.22 % | +7.89 pp | 53.89 % | +20.56 pp | 75.79 % | +42.46 pp |
| Memory Throughput (%) | 12.24 % | 18.60 % | +6.36 pp | 14.64 % | +2.40 pp | 14.57 % | +2.33 pp | 14.67 % | +2.43 pp |
| DRAM Throughput (%) | 8.13 % | 8.12 % | -0.01 pp | 7.76 % | -0.37 pp | 6.86 % | -1.27 pp | 2.66 % | -5.47 pp |
| Achieved Occupancy (%) | 62.13 % | 56.42 % | -5.71 pp | 69.18 % | +7.05 pp | 74.15 % | +12.02 pp | 84.52 % | +22.39 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 24.00 register/thread | +6 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 1.86 | 1.86 | +0.00 | 1.86 | +0.00 | 1.86 | +0.00 | 7.42 | +5.56 |
| No Eligible (%) | 41.43 % | 29.28 % | -12.15 pp | 21.87 % | -19.56 pp | 19.53 % | -21.90 pp | 13.28 % | -28.15 pp |
| IPC Active | 2.43 inst/cycle | 2.73 inst/cycle | +0.30 | 3.08 inst/cycle | +0.65 | 3.22 inst/cycle | +0.79 | 3.50 inst/cycle | +1.07 |

---

## Iteration 5

**Best so far**: 1.0327× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_increase_num_warps_for_occupancy | 1.015× | 1.020× | ✓ |  |
| B | fix_grid_lambda_to_activate_autotune_and_fuse_rsqrt | 1.038× | 0.988× | ✓ |  |
| C | replace_division_with_rsqrt_to_enable_fma_fusion | 1.051× | 1.033× | ✓ | **✓ best** |
| D | fuse_fp32_instructions_via_rsqrt_and_fma | 1.036× | 0.542× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_grid_lambda_t... | Δ | replace_division_... | Δ | fuse_fp32_instruc... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5060 ns | 4960 ns | -100 ns (-2.0%) | 5120 ns | +60 ns (+1.2%) | 4900 ns | -160 ns (-3.2%) | 9340 ns | +4280 ns (+84.6%) |
| SM Throughput (%) | 33.33 % | 12.13 % | -21.20 pp | 22.67 % | -10.66 pp | 30.08 % | -3.25 pp | 63.65 % | +30.32 pp |
| Memory Throughput (%) | 12.24 % | 12.49 % | +0.25 pp | 16.51 % | +4.27 pp | 18.94 % | +6.70 pp | 25.34 % | +13.10 pp |
| DRAM Throughput (%) | 8.13 % | 8.32 % | +0.19 pp | 8.03 % | -0.10 pp | 8.45 % | +0.32 pp | 4.41 % | -3.72 pp |
| Achieved Occupancy (%) | 62.13 % | 49.91 % | -12.22 pp | 52.64 % | -9.49 pp | 55.67 % | -6.46 pp | 75.54 % | +13.41 pp |
| Registers/Thread | 18.00 register/thread | 22.00 register/thread | +4 | 22.00 register/thread | +4 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 1.86 | 0.93 | -0.93 | 1.86 | +0.00 | 1.86 | +0.00 | 7.42 | +5.56 |
| No Eligible (%) | 41.43 % | 59.86 % | +18.43 pp | 51.47 % | +10.04 pp | 29.90 % | -11.53 pp | 20.00 % | -21.43 pp |
| IPC Active | 2.43 inst/cycle | 1.42 inst/cycle | -1.01 | 2.14 inst/cycle | -0.29 | 2.72 inst/cycle | +0.29 | 3.22 inst/cycle | +0.79 |

---

## Cumulative Progress

```
Baseline:  5060 ns  (1.000×)
Iter 2     4900 ns  (1.033×)  — enable_fma_fusion_via_rsqrt_and_fix_grid_lambda

Final best: 4900 ns  (1.0327×)
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
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 501760
@@ -21,19 +33,12 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
+    tmp6 = libdevice.rsqrt(tmp5)
+    tmp11 = tmp2 * tmp6
+    tmp15 = libdevice.fma(tmp11, tmp12, tmp14)
     tmp16 = tl.full([1], 0, tl.int32)
     tmp17 = tl.maximum(tmp16, tmp15)
     tl.store(in_out_ptr0 + (x2), tmp17, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -43,7 +48,7 @@

     arg_4 = inputs[4]
     xnumel = 501760
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 501760, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 501760)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
