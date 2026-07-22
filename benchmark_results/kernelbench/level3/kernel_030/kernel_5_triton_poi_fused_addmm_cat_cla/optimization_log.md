# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_cat_clamp_min_clone_div_expand_linalg_vector_norm_permute_select_transpose_view_zeros_like_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 15580 ns — **Best latency**: 8220 ns (**1.8954×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fuse_fp32_instructions_via_rsqrt_multiply_instead_of_sqrt_divide`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_cat_clamp_min_clone_div_expand_linalg_vector_norm_permute_select_transpose_view_zeros_like_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_030/kernel_5_triton_poi_fused_addmm_cat_cla/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3920,1,1), Waves/SM: 2.47
- **Registers/thread**: 40
- **Duration**: 15580 ns
- **SM Throughput**: 61.35 % | **Memory**: 54.24 % | **DRAM**: 16.65 %
- **Achieved Occupancy**: 61.01 %
- **IPC Active**: 2.93 inst/cycle | **No Eligible**: 24.57 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_tail_wave_and_uncoalesced_stores_via_tile_reshape | 0.988× | 0.996× | ✓ |  |
| B | fix_tile_sizes_and_reduce_register_pressure_via_autotune | 2.037× | 1.316× | ✓ | **✓ best** |
| D | autotune_tile_sizes_fix_uncoalesced_stores | 1.743× | 1.309× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_tail_wave_... | Δ | fix_tile_sizes_an... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 15580 ns | 15650 ns | +70 ns (+0.4%) | 11840 ns | -3740 ns (-24.0%) | 11900 ns | -3680 ns (-23.6%) |
| SM Throughput (%) | 61.35 % | 61.06 % | -0.29 pp | 67.02 % | +5.67 pp | 67.64 % | +6.29 pp |
| Memory Throughput (%) | 54.24 % | 54.17 % | -0.07 pp | 39.73 % | -14.51 pp | 40.14 % | -14.10 pp |
| DRAM Throughput (%) | 16.65 % | 16.61 % | -0.04 pp | 7.57 % | -9.08 pp | 7.54 % | -9.11 pp |
| Achieved Occupancy (%) | 61.01 % | 61.07 % | +0.06 pp | 62.80 % | +1.79 pp | 62.63 % | +1.62 pp |
| Registers/Thread | 40.00 register/thread | 40.00 register/thread | +0 | 34.00 register/thread | -6 | 34.00 register/thread | -6 |
| Waves/SM | 2.47 | 2.47 | +0.00 | 2.47 | +0.00 | 2.47 | +0.00 |
| No Eligible (%) | 24.57 % | 26.20 % | +1.63 pp | 17.24 % | -7.33 pp | 16.79 % | -7.78 pp |
| IPC Active | 2.93 inst/cycle | 2.97 inst/cycle | +0.04 | 3.29 inst/cycle | +0.36 | 3.32 inst/cycle | +0.39 |

---

## Iteration 2

**Best so far**: 1.3159× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_autotune_configs_larger_yblock_and_num_stages_to_eliminate_tail_wave | 1.994× | 1.305× | ✓ |  |
| B | fix_hardcoded_grid_to_enable_autotune_wave_reduction | 1.733× | 1.302× | ✓ |  |
| C | increase_yblock_tile_and_fix_oversized_xblock_via_autotune | 1.505× | 1.302× | ✓ |  |
| D | add_triton_autotune | 2.041× | 1.302× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | expand_autotune_c... | Δ | fix_hardcoded_gri... | Δ | increase_yblock_t... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 15580 ns | 11940 ns | -3640 ns (-23.4%) | 11970 ns | -3610 ns (-23.2%) | 11970 ns | -3610 ns (-23.2%) | 11970 ns | -3610 ns (-23.2%) |
| SM Throughput (%) | 61.35 % | 67.35 % | +6.00 pp | 66.73 % | +5.38 pp | 67.72 % | +6.37 pp | 66.93 % | +5.58 pp |
| Memory Throughput (%) | 54.24 % | 39.96 % | -14.28 pp | 39.58 % | -14.66 pp | 40.15 % | -14.09 pp | 39.69 % | -14.55 pp |
| DRAM Throughput (%) | 16.65 % | 7.54 % | -9.11 pp | 7.50 % | -9.15 pp | 7.51 % | -9.14 pp | 7.52 % | -9.13 pp |
| Achieved Occupancy (%) | 61.01 % | 62.65 % | +1.64 pp | 62.75 % | +1.74 pp | 62.68 % | +1.67 pp | 62.74 % | +1.73 pp |
| Registers/Thread | 40.00 register/thread | 34.00 register/thread | -6 | 34.00 register/thread | -6 | 34.00 register/thread | -6 | 34.00 register/thread | -6 |
| Waves/SM | 2.47 | 2.47 | +0.00 | 2.47 | +0.00 | 2.47 | +0.00 | 2.47 | +0.00 |
| No Eligible (%) | 24.57 % | 16.86 % | -7.71 pp | 16.82 % | -7.75 pp | 16.81 % | -7.76 pp | 17.67 % | -6.90 pp |
| IPC Active | 2.93 inst/cycle | 3.31 inst/cycle | +0.38 | 3.29 inst/cycle | +0.36 | 3.31 inst/cycle | +0.38 | 3.29 inst/cycle | +0.36 |

---

## Iteration 3

**Best so far**: 1.3159× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_enable_autotune_and_expand_configs_for_tail_wave_elimination | 1.985× | 1.316× | ✓ |  |
| B | fix_hardcoded_grid_and_expand_autotune_configs_for_coalesced_stores | 1.770× | 1.305× | ✓ |  |
| C | fix_kernel_function_grid_to_use_lambda_meta_for_autotune | 1.773× | 1.309× | ✓ |  |
| D | add_num_stages_software_pipelining_to_autotune_configs | 2.039× | 1.305× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ | fix_kernel_functi... | Δ | add_num_stages_so... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 15580 ns | 11840 ns | -3740 ns (-24.0%) | 11940 ns | -3640 ns (-23.4%) | 11900 ns | -3680 ns (-23.6%) | 11940 ns | -3640 ns (-23.4%) |
| SM Throughput (%) | 61.35 % | 67.30 % | +5.95 pp | 67.07 % | +5.72 pp | 66.97 % | +5.62 pp | 67.49 % | +6.14 pp |
| Memory Throughput (%) | 54.24 % | 39.89 % | -14.35 pp | 39.80 % | -14.44 pp | 39.73 % | -14.51 pp | 40.02 % | -14.22 pp |
| DRAM Throughput (%) | 16.65 % | 7.58 % | -9.07 pp | 7.53 % | -9.12 pp | 7.55 % | -9.10 pp | 7.53 % | -9.12 pp |
| Achieved Occupancy (%) | 61.01 % | 62.63 % | +1.62 pp | 62.65 % | +1.64 pp | 62.62 % | +1.61 pp | 62.73 % | +1.72 pp |
| Registers/Thread | 40.00 register/thread | 34.00 register/thread | -6 | 34.00 register/thread | -6 | 34.00 register/thread | -6 | 34.00 register/thread | -6 |
| Waves/SM | 2.47 | 2.47 | +0.00 | 2.47 | +0.00 | 2.47 | +0.00 | 2.47 | +0.00 |
| No Eligible (%) | 24.57 % | 16.65 % | -7.92 pp | 17.08 % | -7.49 pp | 18.26 % | -6.31 pp | 16.68 % | -7.89 pp |
| IPC Active | 2.93 inst/cycle | 3.32 inst/cycle | +0.39 | 3.31 inst/cycle | +0.38 | 3.32 inst/cycle | +0.39 | 3.29 inst/cycle | +0.36 |


### Key Learnings

#### ✅ Plan D: `add_num_stages_software_pipelining_to_autotune_configs` (1.305×)

Expanding the autotune search space to include num_stages={1,2,3} alongside existing tile configurations allows Triton's autotuner to select a configuration with software pipelining enabled. However, for pointwise kernels (no tl.dot, no inner reduction loop), the NCU profiling metrics show virtually no change in key signals (Duration, No-Eligible stall %, Eligible Warps/Scheduler, L1/L2 hit rates) between before and after — the 1.30x speedup is likely attributable to the autotuner selecting a marginally better tile shape from the expanded config space rather than software pipelining itself. In pointwise kernels, num_stages has limited effect because there is no multi-iteration loop body for the compiler to pipeline across; the benefit is primarily seen in kernels with inner loops (reductions, GEMMs, convolutions).

> **Applicable when**: Kernel has an inner loop (reduction, GEMM, convolution) with multiple global load instructions per iteration; NCU shows No-Eligible warp stall >15% and Eligible Warps/Scheduler <4; existing autotune configs all lack explicit num_stages; kernel is memory-latency bound at the warp scheduler level.

> **Avoid when**: Kernel is purely pointwise (no inner loop, single pass over data) — in this case num_stages has negligible effect on NCU metrics (Duration, stall %, hit rates all unchanged); the speedup from expanding the config space is marginal and comes from tile shape selection, not pipelining. Do not expect >5% improvement from num_stages alone on pointwise kernels.

---

## Iteration 4

**Best so far**: 1.3159× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_and_add_large_yblock_configs_to_eliminate_tail_wave | 2.082× | 1.305× | ✓ |  |
| C | fix_grid_lambda_and_add_register_reducing_configs_for_full_occupancy | 1.770× | 1.298× | ✓ |  |
| D | fix_grid_lambda_and_increase_tile_sizes_to_eliminate_tail_wave | 2.509× | 1.309× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_grid_lambda_a... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 15580 ns | 11940 ns | -3640 ns (-23.4%) | 12000 ns | -3580 ns (-23.0%) | 11900 ns | -3680 ns (-23.6%) |
| SM Throughput (%) | 61.35 % | 67.30 % | +5.95 pp | 67.48 % | +6.13 pp | 67.79 % | +6.44 pp |
| Memory Throughput (%) | 54.24 % | 39.89 % | -14.35 pp | 40.01 % | -14.23 pp | 40.19 % | -14.05 pp |
| DRAM Throughput (%) | 16.65 % | 7.52 % | -9.13 pp | 7.48 % | -9.17 pp | 7.55 % | -9.10 pp |
| Achieved Occupancy (%) | 61.01 % | 62.68 % | +1.67 pp | 62.71 % | +1.70 pp | 62.73 % | +1.72 pp |
| Registers/Thread | 40.00 register/thread | 34.00 register/thread | -6 | 34.00 register/thread | -6 | 34.00 register/thread | -6 |
| Waves/SM | 2.47 | 2.47 | +0.00 | 2.47 | +0.00 | 2.47 | +0.00 |
| No Eligible (%) | 24.57 % | 16.51 % | -8.06 pp | 16.80 % | -7.77 pp | 16.88 % | -7.69 pp |
| IPC Active | 2.93 inst/cycle | 3.31 inst/cycle | +0.38 | 3.30 inst/cycle | +0.37 | 3.29 inst/cycle | +0.36 |

---

## Iteration 5

**Best so far**: 1.3159× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_instructions_via_rsqrt_multiply_instead_of_sqrt_divide | 2.546× | 1.895× | ✓ | **✓ best** |
| B | fix_hardcoded_grid_lambda_to_activate_autotune_tile_selection | 1.770× | 1.305× | ✓ |  |
| C | fix_hardcoded_grid_and_add_low_register_configs_for_full_occupancy | 2.173× | 1.309× | ✓ |  |
| D | add_triton_autotune | 2.050× | 1.316× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_fp32_instruc... | Δ | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 15580 ns | 8220 ns | -7360 ns (-47.2%) | 11940 ns | -3640 ns (-23.4%) | 11900 ns | -3680 ns (-23.6%) | 11840 ns | -3740 ns (-24.0%) |
| SM Throughput (%) | 61.35 % | 48.13 % | -13.22 pp | 67.39 % | +6.04 pp | 67.86 % | +6.51 pp | 67.63 % | +6.28 pp |
| Memory Throughput (%) | 54.24 % | 55.53 % | +1.29 pp | 39.95 % | -14.29 pp | 40.22 % | -14.02 pp | 40.10 % | -14.14 pp |
| DRAM Throughput (%) | 16.65 % | 10.90 % | -5.75 pp | 7.53 % | -9.12 pp | 7.55 % | -9.10 pp | 7.57 % | -9.08 pp |
| Achieved Occupancy (%) | 61.01 % | 81.40 % | +20.39 pp | 62.75 % | +1.74 pp | 62.68 % | +1.67 pp | 62.70 % | +1.69 pp |
| Registers/Thread | 40.00 register/thread | 32.00 register/thread | -8 | 34.00 register/thread | -6 | 34.00 register/thread | -6 | 34.00 register/thread | -6 |
| Waves/SM | 2.47 | 1.86 | -0.61 | 2.47 | +0.00 | 2.47 | +0.00 | 2.47 | +0.00 |
| No Eligible (%) | 24.57 % | 33.50 % | +8.93 pp | 16.99 % | -7.58 pp | 16.79 % | -7.78 pp | 17.76 % | -6.81 pp |
| IPC Active | 2.93 inst/cycle | 2.64 inst/cycle | -0.29 | 3.30 inst/cycle | +0.37 | 3.31 inst/cycle | +0.38 | 3.30 inst/cycle | +0.37 |

---

## Cumulative Progress

```
Baseline:  15580 ns  (1.000×)
Iter 1     11840 ns  (1.316×)  — fix_tile_sizes_and_reduce_register_pressure_via_autotune
Iter 5     8220 ns  (1.895×)  — fuse_fp32_instructions_via_rsqrt_multiply_instead_of_sqrt_divide

Final best: 8220 ns  (1.8954×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,48 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # num_stages=1 (baseline)
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'YBLOCK':  8},  num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=4, num_stages=1),
+        # num_stages=2 (prefetch next tile while computing current)
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK':  8},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=4, num_stages=2),
+        # num_stages=3 (deeper pipeline)
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK':  8},  num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=4, num_stages=3),
+        # large YBLOCK configs to reduce tail wave
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 64},  num_warps=2),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_addmm_cat_clamp_min_clone_div_expand_linalg_vector_norm_permute_select_transpose_view_zeros_like_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 31360
@@ -21,14 +63,14 @@

     y4 = yindex
     y0 = (yindex % 49)
     y1 = yindex // 49
-    tmp19 = tl.load(in_ptr2 + (96 + x5 + 288*y4), xmask & ymask, eviction_policy='evict_last')
-    tmp21 = tl.load(in_ptr3 + (x3 + 3*y4), xmask & ymask, eviction_policy='evict_last')
+    tmp19 = tl.load(in_ptr2 + (96 + x5 + 288*y4), mask=xmask & ymask, eviction_policy='evict_last')
+    tmp21 = tl.load(in_ptr3 + (x3 + 3*y4), mask=xmask & ymask, eviction_policy='evict_last')
     tmp0 = 96 + x5
     tmp1 = tl.full([1, 1], 0, tl.int64)
     tmp2 = tmp0 >= tmp1
     tmp3 = tl.full([1, 1], 96, tl.int64)
     tmp4 = tmp0 < tmp3
-    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(96 + x2 + 32*x3, [YBLOCK, XBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
+    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(96 + x2 + 32*x3, [YBLOCK, XBLOCK])), mask=tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
     tmp6 = tmp0 >= tmp3
     tmp7 = tl.full([1, 1], 192, tl.int64)
     tmp8 = tmp0 < tmp7
@@ -39,17 +81,15 @@

     tmp13 = tmp0 >= tmp7
     tmp14 = tl.full([1, 1], 288, tl.int64)
     tmp15 = tmp0 < tmp14
-    tmp16 = tl.load(in_ptr1 + (tl.broadcast_to((-96) + x2 + 32*x3, [YBLOCK, XBLOCK])), tmp13 & xmask & ymask, eviction_policy='evict_last', other=0.0)
+    tmp16 = tl.load(in_ptr1 + (tl.broadcast_to((-96) + x2 + 32*x3, [YBLOCK, XBLOCK])), mask=tmp13 & xmask & ymask, eviction_policy='evict_last', other=0.0)
     tmp17 = tl.where(tmp9, tmp12, tmp16)
     tmp18 = tl.where(tmp4, tmp5, tmp17)
     tmp20 = tmp18 + tmp19
-    tmp22 = tl.sqrt_rn(tmp21)
-    tmp23 = tl.full([1, 1], 1e-12, tl.float32)
-    tmp24 = tl.maximum(tmp22, tmp23)
-    tmp25 = (tmp20 / tmp24)
-    tl.store(out_ptr0 + (y0 + 49*x5 + 4704*y1), tmp25, xmask & ymask)
-
-
+    tmp22_rsqrt = libdevice.rsqrt(tmp21)
+    tmp23_inv = tl.full([1, 1], 1.0 / 1e-12, tl.float32)
+    tmp24_inv = tl.minimum(tmp22_rsqrt, tmp23_inv)
+    tmp25 = tmp20 * tmp24_inv
+    tl.store(out_ptr0 + (y0 + 49*x5 + 4704*y1), tmp25, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -60,7 +100,7 @@

     ynumel = 31360
     xnumel = 96
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_addmm_cat_clamp_min_clone_div_expand_linalg_vector_norm_permute_select_transpose_view_zeros_like_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 31360, 96, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_addmm_cat_clamp_min_clone_div_expand_linalg_vector_norm_permute_select_transpose_view_zeros_like_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 31360, 96)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
