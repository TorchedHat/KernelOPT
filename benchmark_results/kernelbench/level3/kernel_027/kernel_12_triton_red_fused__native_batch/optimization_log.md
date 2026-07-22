# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_mean_relu_12` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 11360 ns — **Best latency**: 22780 ns (**0.4987×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_mean_relu_12`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_027/kernel_12_triton_red_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (112,1,1), Waves/SM: 0.42
- **Registers/thread**: 255
- **Duration**: 11360 ns
- **SM Throughput**: 9.71 % | **Memory**: 46.65 % | **DRAM**: 46.65 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.58 inst/cycle | **No Eligible**: 84.99 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_register_pressure_reduction | 1.554× | 0.906× | ✓ |  |
| B | reduce_register_pressure_via_autotune_tile_sizes | 0.532× | 0.182× | ✓ |  |
| C | autotune_tile_sizes_to_reduce_register_pressure_and_improve_occupancy | 1.651× | 0.492× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | reduce_register_p... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 11360 ns | 12540 ns | +1180 ns (+10.4%) | 62430 ns | +51070 ns (+449.6%) | 23070 ns | +11710 ns (+103.1%) |
| SM Throughput (%) | 9.71 % | 9.16 % | -0.55 pp | 6.03 % | -3.68 pp | 5.56 % | -4.15 pp |
| Memory Throughput (%) | 46.65 % | 42.24 % | -4.41 pp | 8.39 % | -38.26 pp | 22.74 % | -23.91 pp |
| DRAM Throughput (%) | 46.65 % | 42.24 % | -4.41 pp | 8.39 % | -38.26 pp | 22.74 % | -23.91 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.22 % | -0.03 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 26.00 register/thread | -229 | 62.00 register/thread | -193 |
| Waves/SM | 0.42 | 0.42 | +0.00 | 0.05 | -0.37 | 0.11 | -0.31 |
| No Eligible (%) | 84.99 % | 86.17 % | +1.18 pp | 87.38 % | +2.39 pp | 92.89 % | +7.90 pp |
| IPC Active | 0.58 inst/cycle | 0.54 inst/cycle | -0.04 | 0.50 inst/cycle | -0.08 | 0.28 inst/cycle | -0.30 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_dynamic_grid_fix | 1.557× | 0.936× | ✓ |  |
| B | fix_static_grid_and_add_vectorized_load_hints | 1.640× | 0.487× | ✓ |  |
| C | fix_autotune_grid_lambda_and_add_vectorized_load_hints | 0.532× | 0.184× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fix_static_grid_a... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 11360 ns | 12130 ns | +770 ns (+6.8%) | 23330 ns | +11970 ns (+105.4%) | 61760 ns | +50400 ns (+443.7%) |
| SM Throughput (%) | 9.71 % | 9.34 % | -0.37 pp | 5.54 % | -4.17 pp | 6.05 % | -3.66 pp |
| Memory Throughput (%) | 46.65 % | 43.77 % | -2.88 pp | 22.48 % | -24.17 pp | 8.49 % | -38.16 pp |
| DRAM Throughput (%) | 46.65 % | 43.77 % | -2.88 pp | 22.48 % | -24.17 pp | 8.49 % | -38.16 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.21 % | -0.04 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 62.00 register/thread | -193 | 26.00 register/thread | -229 |
| Waves/SM | 0.42 | 0.42 | +0.00 | 0.11 | -0.31 | 0.05 | -0.37 |
| No Eligible (%) | 84.99 % | 86.25 % | +1.26 pp | 92.87 % | +7.88 pp | 87.35 % | +2.36 pp |
| IPC Active | 0.58 inst/cycle | 0.56 inst/cycle | -0.02 | 0.28 inst/cycle | -0.30 | 0.50 inst/cycle | -0.08 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_lambda_and_remove_broken_list_hints | 1.558× | 0.927× | ✓ |  |
| B | fix_autotune_grid_lambda_to_restore_block_count | 1.865× | 0.499× | ✓ | **✓ best** |
| C | fix_static_grid_and_restore_large_r0_block_configs | 1.861× | 0.796× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_l... | Δ | fix_autotune_grid... | Δ | fix_static_grid_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 11360 ns | 12260 ns | +900 ns (+7.9%) | 22780 ns | +11420 ns (+100.5%) | 14270 ns | +2910 ns (+25.6%) |
| SM Throughput (%) | 9.71 % | 9.36 % | -0.35 pp | 5.54 % | -4.17 pp | 16.13 % | +6.42 pp |
| Memory Throughput (%) | 46.65 % | 43.32 % | -3.33 pp | 23.01 % | -23.64 pp | 37.71 % | -8.94 pp |
| DRAM Throughput (%) | 46.65 % | 43.32 % | -3.33 pp | 23.01 % | -23.64 pp | 37.71 % | -8.94 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 12.48 % | +6.23 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 62.00 register/thread | -193 | 255.00 register/thread | +0 |
| Waves/SM | 0.42 | 0.42 | +0.00 | 0.11 | -0.31 | 0.85 | +0.43 |
| No Eligible (%) | 84.99 % | 86.25 % | +1.26 pp | 92.99 % | +8.00 pp | 68.79 % | -16.20 pp |
| IPC Active | 0.58 inst/cycle | 0.54 inst/cycle | -0.04 | 0.28 inst/cycle | -0.30 | 1.23 inst/cycle | +0.65 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_lambda_and_cap_register_pressure_with_maxnreg | 1.589× | 0.910× | ✓ |  |
| B | fix_static_grid_lambda_and_restore_num_stages_2_configs | 1.715× | 1.017× | ✓ |  |
| C | fix_static_grid_to_lambda_for_autotune_xblock_scaling | 1.813× | 0.492× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_l... | Δ | fix_static_grid_l... | Δ | fix_static_grid_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 11360 ns | 12480 ns | +1120 ns (+9.9%) | 11170 ns | -190 ns (-1.7%) | 23100 ns | +11740 ns (+103.3%) |
| SM Throughput (%) | 9.71 % | 9.41 % | -0.30 pp | 9.29 % | -0.42 pp | 5.56 % | -4.15 pp |
| Memory Throughput (%) | 46.65 % | 42.54 % | -4.11 pp | 47.18 % | +0.53 pp | 22.72 % | -23.93 pp |
| DRAM Throughput (%) | 46.65 % | 42.54 % | -4.11 pp | 47.18 % | +0.53 pp | 22.72 % | -23.93 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 12.48 % | +6.23 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 90.00 register/thread | -165 | 62.00 register/thread | -193 |
| Waves/SM | 0.42 | 0.42 | +0.00 | 0.42 | +0.00 | 0.11 | -0.31 |
| No Eligible (%) | 84.99 % | 86.12 % | +1.13 pp | 85.78 % | +0.79 pp | 92.84 % | +7.85 pp |
| IPC Active | 0.58 inst/cycle | 0.56 inst/cycle | -0.02 | 0.54 inst/cycle | -0.04 | 0.28 inst/cycle | -0.30 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_lambda_to_unlock_autotune_xblock_scaling | 1.710× | 0.995× | ✓ |  |
| C | fix_static_grid_to_lambda_for_correct_autotune_block_count | 1.829× | 0.495× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_l... | Δ | fix_static_grid_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 11360 ns | 11420 ns | +60 ns (+0.5%) | 22940 ns | +11580 ns (+101.9%) |
| SM Throughput (%) | 9.71 % | 9.47 % | -0.24 pp | 5.55 % | -4.16 pp |
| Memory Throughput (%) | 46.65 % | 45.92 % | -0.73 pp | 22.87 % | -23.78 pp |
| DRAM Throughput (%) | 46.65 % | 45.92 % | -0.73 pp | 22.87 % | -23.78 pp |
| Achieved Occupancy (%) | 6.25 % | 12.48 % | +6.23 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 90.00 register/thread | -165 | 62.00 register/thread | -193 |
| Waves/SM | 0.42 | 0.42 | +0.00 | 0.11 | -0.31 |
| No Eligible (%) | 84.99 % | 86.75 % | +1.76 pp | 92.92 % | +7.93 pp |
| IPC Active | 0.58 inst/cycle | 0.57 inst/cycle | -0.01 | 0.28 inst/cycle | -0.30 |

---

## Cumulative Progress

```
Baseline:  11360 ns  (1.000×)

Final best: 22780 ns  (0.4987×)
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
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 64},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 64},  num_warps=8, num_stages=2),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_mean_relu_12(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 14336
@@ -16,7 +28,7 @@

     xmask = xindex < xnumel
     r0_base = tl.arange(0, R0_BLOCK)[None, :]
     rbase = r0_base
-    x0 = (xindex % 256)
+    x0 = tl.max_contiguous(tl.multiple_of((xindex % 256), [256, 1]), [XBLOCK, 1])
     x1 = xindex // 256
     _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
     x3 = xindex
@@ -26,10 +38,10 @@

         roffset = r0_offset
         rindex = r0_index
         r0_2 = r0_index
-        tmp0 = tl.load(in_ptr0 + (x0 + 512*((r0_2 % 28)) + 28672*(r0_2 // 28) + 114688*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-        tmp1 = tl.load(in_ptr0 + (256 + x0 + 512*((r0_2 % 28)) + 28672*(r0_2 // 28) + 114688*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-        tmp3 = tl.load(in_ptr0 + (14336 + x0 + 512*((r0_2 % 28)) + 28672*(r0_2 // 28) + 114688*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-        tmp5 = tl.load(in_ptr0 + (14592 + x0 + 512*((r0_2 % 28)) + 28672*(r0_2 // 28) + 114688*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp0 = tl.load(in_ptr0 + (x0 + 512*((r0_2 % 28)) + 28672*(r0_2 // 28) + 114688*x1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp1 = tl.load(in_ptr0 + (256 + x0 + 512*((r0_2 % 28)) + 28672*(r0_2 // 28) + 114688*x1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp3 = tl.load(in_ptr0 + (14336 + x0 + 512*((r0_2 % 28)) + 28672*(r0_2 // 28) + 114688*x1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp5 = tl.load(in_ptr0 + (14592 + x0 + 512*((r0_2 % 28)) + 28672*(r0_2 // 28) + 114688*x1), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
         tmp2 = tl.maximum(tmp0, tmp1)
         tmp4 = tl.maximum(tmp2, tmp3)
         tmp6 = tl.maximum(tmp4, tmp5)
@@ -37,9 +49,7 @@

         tmp9 = _tmp8 + tmp7
         _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
     tmp8 = tl.sum(_tmp8, 1)[:, None]
-    tl.store(out_ptr0 + (x3), tmp8, xmask)
-
-
+    tl.store(out_ptr0 + (x3), tmp8, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -47,7 +57,7 @@

     xnumel = 14336
     r0_numel = 112
     grid = (triton.cdiv(xnumel, 128),)
-    triton_red_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_mean_relu_12[grid](arg_0, arg_1, 14336, 112, XBLOCK=128, R0_BLOCK=512)
+    triton_red_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_mean_relu_12[grid](arg_0, arg_1, 14336, 112)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
