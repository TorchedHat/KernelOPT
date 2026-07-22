# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3940 ns — **Best latency**: 3070 ns (**1.2834×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_launcher_grid_lambda_and_expand_autotune_configs`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_37_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1792,1,1), Waves/SM: 0.85
- **Registers/thread**: 28
- **Duration**: 3940 ns
- **SM Throughput**: 25.60 % | **Memory**: 23.52 % | **DRAM**: 2.72 %
- **Achieved Occupancy**: 56.98 %
- **IPC Active**: 2.09 inst/cycle | **No Eligible**: 46.48 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | fix_tile_shape_and_add_autotune_for_vectorized_access | 1.066× | 0.826× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_tile_shape_an... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3940 ns | 4770 ns | +830 ns (+21.1%) |
| SM Throughput (%) | 25.60 % | 43.75 % | +18.15 pp |
| Memory Throughput (%) | 23.52 % | 36.36 % | +12.84 pp |
| DRAM Throughput (%) | 2.72 % | 2.25 % | -0.47 pp |
| Achieved Occupancy (%) | 56.98 % | 56.26 % | -0.72 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 |
| Waves/SM | 0.85 | 0.85 | +0.00 |
| No Eligible (%) | 46.48 % | 23.29 % | -23.19 pp |
| IPC Active | 2.09 inst/cycle | 3.00 inst/cycle | +0.91 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_correct_meta_grid | 1.111× | 0.821× | ✓ |  |
| B | fix_autotune_grid_lambda_to_eliminate_instruction_bloat | 1.102× | 0.826× | ✓ |  |
| C | fix_autotune_grid_lambda_to_use_meta_tile_sizes | 1.113× | 0.831× | ✓ |  |
| D | fix_autotune_grid_lambda_and_coalescing_configs | 1.109× | 1.243× | ✓ |  |
| E | add_triton_autotune | 1.114× | 0.844× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3940 ns | 4800 ns | +860 ns (+21.8%) | 4770 ns | +830 ns (+21.1%) | 4740 ns | +800 ns (+20.3%) | 3170 ns | -770 ns (-19.5%) | 4670 ns | +730 ns (+18.5%) |
| SM Throughput (%) | 25.60 % | 43.52 % | +17.92 pp | 45.24 % | +19.64 pp | 45.60 % | +20.00 pp | 8.78 % | -16.82 pp | 45.30 % | +19.70 pp |
| Memory Throughput (%) | 23.52 % | 36.17 % | +12.65 pp | 37.57 % | +14.05 pp | 37.87 % | +14.35 pp | 3.39 % | -20.13 pp | 37.72 % | +14.20 pp |
| DRAM Throughput (%) | 2.72 % | 2.23 % | -0.49 pp | 2.25 % | -0.47 pp | 2.26 % | -0.46 pp | 3.39 % | +0.67 pp | 2.29 % | -0.43 pp |
| Achieved Occupancy (%) | 56.98 % | 56.13 % | -0.85 pp | 56.07 % | -0.91 pp | 56.05 % | -0.93 pp | 42.46 % | -14.52 pp | 56.57 % | -0.41 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 16.00 register/thread | -12 | 32.00 register/thread | +4 |
| Waves/SM | 0.85 | 0.85 | +0.00 | 0.85 | +0.00 | 0.85 | +0.00 | 0.85 | +0.00 | 0.85 | +0.00 |
| No Eligible (%) | 46.48 % | 22.99 % | -23.49 pp | 26.30 % | -20.18 pp | 23.61 % | -22.87 pp | 63.29 % | +16.81 pp | 24.02 % | -22.46 pp |
| IPC Active | 2.09 inst/cycle | 2.99 inst/cycle | +0.90 | 2.99 inst/cycle | +0.90 | 3.01 inst/cycle | +0.92 | 1.24 inst/cycle | -0.85 | 2.98 inst/cycle | +0.89 |


### Key Learnings

#### ❌ Plan C: `fix_autotune_grid_lambda_to_use_meta_tile_sizes` (0.831×)

Reordering @triton.autotune configs (moving XBLOCK=16 configs to the top of the list) caused the autotuner to select a config with XBLOCK=16 instead of XBLOCK=1, inflating executed instructions from 202,048 to 1,705,984 (8.4x increase) and doubling kernel duration from 3.17 µs to 4.74 µs. The XBLOCK=16 configs also doubled register pressure (16→32 registers/thread) and introduced 8 bytes of dynamic shared memory per block, further degrading performance. When xnumel=9, XBLOCK=16 wastes 7 of 16 threads per x-tile on masked-out lanes, generating far more instructions than XBLOCK=1 which covers xnumel=9 in a single tile. Config ordering in @triton.autotune affects which config is benchmarked first and can bias selection; placing suboptimal configs first can cause the autotuner to converge on a worse solution.

> **Applicable when**: A Triton kernel has xnumel much smaller than XBLOCK (e.g., xnumel=9 with XBLOCK=16), causing most threads in the x-dimension to be masked out. Profiling shows executed instructions 8x higher than expected, doubled register pressure, and unexpected dynamic shared memory allocation after a config reorder.

> **Avoid when**: When xnumel is large enough to fill XBLOCK tiles efficiently (xnumel >> XBLOCK), XBLOCK=16 configs may be beneficial. Do not reorder autotune configs when the current ordering already produces optimal results (low instruction count, low register pressure).

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_tiled_transpose_for_coalesced_stores | 1.078× | 1.309× | ✓ |  |
| B | fix_static_launcher_grid_to_meta_lambda | 1.092× | 0.831× | ✓ |  |
| C | shared_memory_tiled_transpose_for_coalesced_access | 1.115× | 1.309× | ✓ | **✓ best** |
| D | persistent_1d_tile_scheduling_for_transpose_kernel | 0.001× | 0.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_til... | Δ | fix_static_launch... | Δ | shared_memory_til... | Δ | persistent_1d_til... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3940 ns | 3010 ns | -930 ns (-23.6%) | 4740 ns | +800 ns (+20.3%) | 3010 ns | -930 ns (-23.6%) | 7670000 ns | +7666060 ns (+194570.1%) |
| SM Throughput (%) | 25.60 % | 5.66 % | -19.94 pp | 45.48 % | +19.88 pp | 8.99 % | -16.61 pp | 58.16 % | +32.56 pp |
| Memory Throughput (%) | 23.52 % | 3.54 % | -19.98 pp | 37.79 % | +14.27 pp | 3.55 % | -19.97 pp | 24.88 % | +1.36 pp |
| DRAM Throughput (%) | 2.72 % | 3.54 % | +0.82 pp | 2.27 % | -0.45 pp | 3.55 % | +0.83 pp | 0.00 % | -2.72 pp |
| Achieved Occupancy (%) | 56.98 % | 19.44 % | -37.54 pp | 56.15 % | -0.83 pp | 40.27 % | -16.71 pp | 72.69 % | +15.71 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 32.00 register/thread | +4 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.85 | 0.42 | -0.43 | 0.85 | +0.00 | 0.85 | +0.00 | 0.85 | +0.00 |
| No Eligible (%) | 46.48 % | 80.65 % | +34.17 pp | 23.60 % | -22.88 pp | 65.33 % | +18.85 pp | 40.79 % | -5.69 pp |
| IPC Active | 2.09 inst/cycle | 0.73 inst/cycle | -1.36 | 3.00 inst/cycle | +0.91 | 1.21 inst/cycle | -0.88 | 2.40 inst/cycle | +0.31 |


### Key Learnings

#### ❌ Plan D: `persistent_1d_tile_scheduling_for_transpose_kernel` (0.002×)

Converting a 2D Triton transpose kernel to a persistent 1D while-loop but leaving the launcher grid unchanged at the old static value `(cdiv(xnumel, 128), cdiv(ynumel, 8)) = (1, 1792)` caused executed instructions to explode from 202K to 1.23 billion (~6,100x increase) and duration to increase from 3.04 µs to 2.22 ms (~730x slower). The root cause is identical to item c9f1a3b7: the kernel body uses `tl.num_programs(0)` to stride the while-loop, but with the old 2D grid, `tl.program_id(0)=0` for all blocks and `tl.num_programs(0)=1`, so each of the 1792 CTAs iterates over the entire tile space independently instead of cooperatively. The L1 hit rate jumped to 71% and L2 to 97.9% (data fits in cache due to repeated access), and occupancy improved to 70.75%, but the sheer instruction volume (1.23B) dominated all other effects. This is the second occurrence of this exact failure mode: a persistent while-loop kernel MUST have its launcher grid updated to `(NUM_PROGRAMS,)` simultaneously with the kernel body change.

> **Applicable when**: Never apply the broken pattern. The persistent 1D while-loop approach is theoretically valid ONLY when the launcher grid is simultaneously updated to `(NUM_PROGRAMS,)` where NUM_PROGRAMS matches the stride used in the while-loop body.

> **Avoid when**: Do not convert a 2D Triton kernel to a persistent 1D while-loop if the launcher grid is not simultaneously updated to match the new persistent CTA count. This is the third iteration (items c9f1a3b7 and now e5a1c027) where this exact mistake has caused catastrophic regression. Also avoid when the kernel is already fast (<5 µs baseline) — the while-loop control flow overhead (integer division by non-power-of-2 like 9) may exceed any occupancy gains even when implemented correctly.

---

## Iteration 4

**Best so far**: 1.309× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_launcher_grid_meta_lambda_with_large_yblock_coalesced_configs | 1.114× | 1.296× | ✓ |  |
| B | fix_launcher_static_grid_to_meta_lambda_grid | 1.117× | 1.296× | ✓ |  |
| C | fix_launcher_grid_lambda_and_expand_autotune_configs | 1.079× | 1.283× | ✓ |  |
| D | revert_broken_persistent_kernel_and_fix_coalescing_with_aligned_yblock | 1.074× | 1.283× | ✓ |  |
| E | add_triton_autotune | 1.119× | 1.283× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_launcher_grid... | Δ | fix_launcher_stat... | Δ | fix_launcher_grid... | Δ | revert_broken_per... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3940 ns | 3040 ns | -900 ns (-22.8%) | 3040 ns | -900 ns (-22.8%) | 3070 ns | -870 ns (-22.1%) | 3070 ns | -870 ns (-22.1%) | 3070 ns | -870 ns (-22.1%) |
| SM Throughput (%) | 25.60 % | 8.45 % | -17.15 pp | 8.97 % | -16.63 pp | 6.39 % | -19.21 pp | 11.20 % | -14.40 pp | 19.33 % | -6.27 pp |
| Memory Throughput (%) | 23.52 % | 3.54 % | -19.98 pp | 3.53 % | -19.99 pp | 3.52 % | -20.00 pp | 3.49 % | -20.03 pp | 6.98 % | -16.54 pp |
| DRAM Throughput (%) | 2.72 % | 3.54 % | +0.82 pp | 3.53 % | +0.81 pp | 3.52 % | +0.80 pp | 3.49 % | +0.77 pp | 3.47 % | +0.75 pp |
| Achieved Occupancy (%) | 56.98 % | 40.46 % | -16.52 pp | 42.26 % | -14.72 pp | 17.90 % | -39.08 pp | 48.08 % | -8.90 pp | 58.29 % | +1.31 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.85 | 0.85 | +0.00 | 0.85 | +0.00 | 0.42 | -0.43 | 0.85 | +0.00 | 1.70 | +0.85 |
| No Eligible (%) | 46.48 % | 64.00 % | +17.52 pp | 64.48 % | +18.00 pp | 73.08 % | +26.60 pp | 62.53 % | +16.05 pp | 39.75 % | -6.73 pp |
| IPC Active | 2.09 inst/cycle | 1.18 inst/cycle | -0.91 | 1.23 inst/cycle | -0.86 | 0.84 inst/cycle | -1.25 | 1.58 inst/cycle | -0.51 | 2.18 inst/cycle | +0.09 |

---

## Iteration 5

**Best so far**: 1.309× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fix_launcher_static_grid_and_explicit_tile_override_to_enable_autotune | 1.112× | 1.309× | ✓ |  |
| D | persistent_flat1d_kernel_with_correct_sm_grid | 0.004× | 0.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_launcher_stat... | Δ | persistent_flat1d... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3940 ns | 3010 ns | -930 ns (-23.6%) | 2220000 ns | +2216060 ns (+56245.2%) |
| SM Throughput (%) | 25.60 % | 6.05 % | -19.55 pp | 70.03 % | +44.43 pp |
| Memory Throughput (%) | 23.52 % | 3.56 % | -19.96 pp | 85.21 % | +61.69 pp |
| DRAM Throughput (%) | 2.72 % | 3.56 % | +0.84 pp | 0.01 % | -2.71 pp |
| Achieved Occupancy (%) | 56.98 % | 17.61 % | -39.37 pp | 70.75 % | +13.77 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 32.00 register/thread | +4 |
| Waves/SM | 0.85 | 0.42 | -0.43 | 0.85 | +0.00 |
| No Eligible (%) | 46.48 % | 72.48 % | +26.00 pp | 27.62 % | -18.86 pp |
| IPC Active | 2.09 inst/cycle | 0.83 inst/cycle | -1.26 | 2.89 inst/cycle | +0.80 |

---

## Cumulative Progress

```
Baseline:  3940 ns  (1.000×)
Iter 3     3010 ns  (1.309×)  — shared_memory_tiled_transpose_for_coalesced_access

Final best: 3070 ns  (1.2834×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,29 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'YBLOCK': 128,  'XBLOCK': 1},  num_warps=4,  num_stages=1),
+        triton.Config({'YBLOCK': 256,  'XBLOCK': 1},  num_warps=8,  num_stages=1),
+        triton.Config({'YBLOCK': 512,  'XBLOCK': 1},  num_warps=16, num_stages=1),
+        triton.Config({'YBLOCK': 1024, 'XBLOCK': 1},  num_warps=16, num_stages=1),
+        triton.Config({'YBLOCK': 2048, 'XBLOCK': 1},  num_warps=16, num_stages=1),
+        triton.Config({'YBLOCK': 64,   'XBLOCK': 16}, num_warps=4,  num_stages=1),
+        triton.Config({'YBLOCK': 128,  'XBLOCK': 16}, num_warps=4,  num_stages=1),
+        triton.Config({'YBLOCK': 256,  'XBLOCK': 16}, num_warps=8,  num_stages=1),
+        triton.Config({'YBLOCK': 512,  'XBLOCK': 16}, num_warps=16, num_stages=1),
+        triton.Config({'YBLOCK': 1024, 'XBLOCK': 16}, num_warps=16, num_stages=1),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 14336
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +35,8 @@

     y3 = yindex
     y0 = (yindex % 448)
     y1 = yindex // 448
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 448*x2 + 4032*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last', other=0.0)
+    tl.store(out_ptr0 + (y0 + 448*x2 + 4032*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +44,7 @@

     ynumel = 14336
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37[grid](arg_0, arg_1, 14336, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37[grid](arg_0, arg_1, 14336, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
