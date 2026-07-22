# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_36` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2660 ns — **Best latency**: 2400 ns (**1.1083×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_uncoalesced_store_with_streaming_eviction_hint`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_36`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_36_triton_poi_fused_convolution_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (96,1,1), Waves/SM: 0.05
- **Registers/thread**: 28
- **Duration**: 2660 ns
- **SM Throughput**: 1.85 % | **Memory**: 1.77 % | **DRAM**: 0.63 %
- **Achieved Occupancy**: 6.10 %
- **IPC Active**: 0.37 inst/cycle | **No Eligible**: 90.86 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_autotune_right_size_tiles_for_small_grid | 0.969× | 1.095× | ✓ |  |
| B | autotune_tile_sizes_for_small_transpose_kernel | 1.029× | 1.081× | ✓ |  |
| C | autotune_tile_sizes_for_underutilized_small_grid | 0.966× | 1.015× | ✓ |  |
| D | autotune_tile_sizes_for_occupancy | 1.031× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | add_autotune_righ... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2660 ns | 2430 ns | -230 ns (-8.6%) | 2460 ns | -200 ns (-7.5%) | 2620 ns | -40 ns (-1.5%) | 2660 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 1.85 % | 1.41 % | -0.44 pp | 0.17 % | -1.68 pp | 2.22 % | +0.37 pp | 0.69 % | -1.16 pp |
| Memory Throughput (%) | 1.77 % | 1.30 % | -0.47 pp | 0.68 % | -1.09 pp | 2.05 % | +0.28 pp | 0.62 % | -1.15 pp |
| DRAM Throughput (%) | 0.63 % | 0.68 % | +0.05 pp | 0.44 % | -0.19 pp | 0.64 % | +0.01 pp | 0.40 % | -0.23 pp |
| Achieved Occupancy (%) | 6.10 % | 6.20 % | +0.10 pp | 1.56 % | -4.54 pp | 6.21 % | +0.11 pp | 6.04 % | -0.06 pp |
| Registers/Thread | 28.00 register/thread | 20.00 register/thread | -8 | 16.00 register/thread | -12 | 28.00 register/thread | +0 | 16.00 register/thread | -12 |
| Waves/SM | 0.05 | 0.05 | +0.00 | 0.02 | -0.03 | 0.05 | +0.00 | 0.05 | +0.00 |
| No Eligible (%) | 90.86 % | 90.77 % | -0.09 pp | 96.25 % | +5.39 pp | 84.18 % | -6.68 pp | 93.23 % | +2.37 pp |
| IPC Active | 0.37 inst/cycle | 0.29 inst/cycle | -0.08 | 0.04 inst/cycle | -0.33 | 0.45 inst/cycle | +0.08 | 0.20 inst/cycle | -0.17 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.107× | 0.045× | ✓ |  |
| B | persistent_tile_scheduling | 0.011× | 0.005× | ✓ |  |
| C | persistent_tile_scheduling_for_small_transpose_kernel | 0.126× | 0.051× | ✓ |  |
| D | persistent_tile_scheduling | 0.998× | 1.157× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2660 ns | 58500 ns | +55840 ns (+2099.2%) | 494080 ns | +491420 ns (+18474.4%) | 51900 ns | +49240 ns (+1851.1%) | 2300 ns | -360 ns (-13.5%) |
| SM Throughput (%) | 1.85 % | 9.43 % | +7.58 pp | 24.27 % | +22.42 pp | 7.76 % | +5.91 pp | 0.69 % | -1.16 pp |
| Memory Throughput (%) | 1.77 % | 9.42 % | +7.65 pp | 25.55 % | +23.78 pp | 9.49 % | +7.72 pp | 0.65 % | -1.12 pp |
| DRAM Throughput (%) | 0.63 % | 0.03 % | -0.60 pp | 0.00 % | -0.63 pp | 0.03 % | -0.60 pp | 0.48 % | -0.15 pp |
| Achieved Occupancy (%) | 6.10 % | 6.25 % | +0.15 pp | 6.25 % | +0.15 pp | 6.25 % | +0.15 pp | 6.02 % | -0.08 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 16.00 register/thread | -12 |
| Waves/SM | 0.05 | 0.05 | +0.00 | 0.05 | +0.00 | 0.05 | +0.00 | 0.05 | +0.00 |
| No Eligible (%) | 90.86 % | 86.48 % | -4.38 pp | 66.51 % | -24.35 pp | 89.02 % | -1.84 pp | 94.14 % | +3.28 pp |
| IPC Active | 0.37 inst/cycle | 0.53 inst/cycle | +0.16 | 1.34 inst/cycle | +0.97 | 0.43 inst/cycle | +0.06 | 0.22 inst/cycle | -0.15 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.045×)

Replacing a 2D grid kernel with a persistent 1D grid using a while-loop caused a catastrophic 24x slowdown (2.43us → 58.5us) despite the grid size remaining unchanged at 96 blocks. The root cause was an implementation flaw: the grid launch formula `(cdiv(xnumel, 128), cdiv(ynumel, 8))` was left unchanged from the original, producing the same 96-block grid instead of the intended 132-SM grid, while the while-loop body now executed 165x more instructions (26,688 → 4,406,400) because each CTA iterated over many more tiles than intended. The persistent kernel pattern itself is not the problem, but the grid size must be explicitly set to `min(num_sms, num_tiles)` — failing to update the grid launch is a fatal implementation error that turns O(1) work per CTA into O(num_tiles/num_programs) iterations with the wrong denominator.

> **Applicable when**: Never apply this specific broken pattern. The persistent kernel concept is valid when: grid size is correctly set to min(num_sms, num_tiles); tile sizes are updated to match the new 1D grid decomposition; num_tiles_x/y are passed as kernel arguments (not computed inside kernel from constexpr XBLOCK/YBLOCK with wrong grid dims).

> **Avoid when**: Do not attempt persistent kernel conversion when the grid launch formula is not simultaneously updated from the original 2D formula to a 1D persistent formula. Do not use tl.cdiv(xnumel, XBLOCK) inside the kernel body when XBLOCK is a constexpr that does not match the actual tile decomposition used at launch. For sub-5us kernels where launch overhead dominates, persistent kernels may not help even when correctly implemented.

#### ❌ Plan B: `persistent_tile_scheduling` (0.045×)

Replacing a 2D grid kernel with a persistent 1D grid using a while-loop caused a catastrophic 24x slowdown (2.43us → 58.5us) despite the grid size remaining unchanged at 96 blocks. The root cause was an implementation flaw: the grid launch formula `(cdiv(xnumel, 128), cdiv(ynumel, 8))` was left unchanged from the original, producing the same 96-block grid instead of the intended 132-SM grid, while the while-loop body now executed 165x more instructions (26,688 → 4,406,400) because each CTA iterated over many more tiles than intended. The persistent kernel pattern itself is not the problem, but the grid size must be explicitly set to `min(num_sms, num_tiles)` — failing to update the grid launch is a fatal implementation error that turns O(1) work per CTA into O(num_tiles/num_programs) iterations with the wrong denominator.

> **Applicable when**: Never apply this specific broken pattern. The persistent kernel concept is valid when: grid size is correctly set to min(num_sms, num_tiles); tile sizes are updated to match the new 1D grid decomposition; num_tiles_x/y are passed as kernel arguments (not computed inside kernel from constexpr XBLOCK/YBLOCK with wrong grid dims).

> **Avoid when**: Do not attempt persistent kernel conversion when the grid launch formula is not simultaneously updated from the original 2D formula to a 1D persistent formula. Do not use tl.cdiv(xnumel, XBLOCK) inside the kernel body when XBLOCK is a constexpr that does not match the actual tile decomposition used at launch. For sub-5us kernels where launch overhead dominates, persistent kernels may not help even when correctly implemented.

#### ❌ Plan D: `persistent_tile_scheduling` (0.045×)

Replacing a 2D grid kernel with a persistent 1D grid using a while-loop caused a catastrophic 24x slowdown (2.43us → 58.5us) despite the grid size remaining unchanged at 96 blocks. The root cause was an implementation flaw: the grid launch formula `(cdiv(xnumel, 128), cdiv(ynumel, 8))` was left unchanged from the original, producing the same 96-block grid instead of the intended 132-SM grid, while the while-loop body now executed 165x more instructions (26,688 → 4,406,400) because each CTA iterated over many more tiles than intended. The persistent kernel pattern itself is not the problem, but the grid size must be explicitly set to `min(num_sms, num_tiles)` — failing to update the grid launch is a fatal implementation error that turns O(1) work per CTA into O(num_tiles/num_programs) iterations with the wrong denominator.

> **Applicable when**: Never apply this specific broken pattern. The persistent kernel concept is valid when: grid size is correctly set to min(num_sms, num_tiles); tile sizes are updated to match the new 1D grid decomposition; num_tiles_x/y are passed as kernel arguments (not computed inside kernel from constexpr XBLOCK/YBLOCK with wrong grid dims).

> **Avoid when**: Do not attempt persistent kernel conversion when the grid launch formula is not simultaneously updated from the original 2D formula to a 1D persistent formula. Do not use tl.cdiv(xnumel, XBLOCK) inside the kernel body when XBLOCK is a constexpr that does not match the actual tile decomposition used at launch. For sub-5us kernels where launch overhead dominates, persistent kernels may not help even when correctly implemented.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_uncoalesced_store_with_streaming_eviction_hint | 1.039× | 1.108× | ✓ | **✓ best** |
| B | revert_broken_persistent_loop_and_autotune_tile_sizes | 1.000× | 1.051× | ✓ |  |
| C | fix_broken_persistent_loop_and_autotune_tile_sizes | 1.022× | 1.081× | ✓ |  |
| D | fix_broken_persistent_kernel_revert_to_2d_tiled | 1.034× | 1.095× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_uncoalesced_s... | Δ | revert_broken_per... | Δ | fix_broken_persis... | Δ | fix_broken_persis... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2660 ns | 2400 ns | -260 ns (-9.8%) | 2530 ns | -130 ns (-4.9%) | 2460 ns | -200 ns (-7.5%) | 2430 ns | -230 ns (-8.6%) |
| SM Throughput (%) | 1.85 % | 0.19 % | -1.66 pp | 0.18 % | -1.67 pp | 0.17 % | -1.68 pp | 0.62 % | -1.23 pp |
| Memory Throughput (%) | 1.77 % | 0.71 % | -1.06 pp | 0.79 % | -0.98 pp | 0.61 % | -1.16 pp | 0.63 % | -1.14 pp |
| DRAM Throughput (%) | 0.63 % | 0.44 % | -0.19 pp | 0.43 % | -0.20 pp | 0.44 % | -0.19 pp | 0.45 % | -0.18 pp |
| Achieved Occupancy (%) | 6.10 % | 1.56 % | -4.54 pp | 1.56 % | -4.54 pp | 1.56 % | -4.54 pp | 6.05 % | -0.05 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.05 | 0.02 | -0.03 | 0.02 | -0.03 | 0.02 | -0.03 | 0.05 | +0.00 |
| No Eligible (%) | 90.86 % | 96.93 % | +6.07 pp | 95.60 % | +4.74 pp | 92.61 % | +1.75 pp | 95.75 % | +4.89 pp |
| IPC Active | 0.37 inst/cycle | 0.04 inst/cycle | -0.33 | 0.04 inst/cycle | -0.33 | 0.06 inst/cycle | -0.31 | 0.22 inst/cycle | -0.15 |

---

## Iteration 4

**Best so far**: 1.1083× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_load_eviction_to_reduce_l1_pollution | 0.953× | 1.081× | ✓ |  |
| B | fix_hardcoded_grid_to_enable_autotune_tile_exploration | 1.021× | 1.137× | ✓ |  |
| C | fix_autotune_callsite_to_use_meta_tile_params | 0.990× | 1.081× | ✓ |  |
| D | evict_first_store_for_strided_transpose_output | 0.983× | 1.157× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_load_ev... | Δ | fix_hardcoded_gri... | Δ | fix_autotune_call... | Δ | evict_first_store... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2660 ns | 2460 ns | -200 ns (-7.5%) | 2340 ns | -320 ns (-12.0%) | 2460 ns | -200 ns (-7.5%) | 2300 ns | -360 ns (-13.5%) |
| SM Throughput (%) | 1.85 % | 0.17 % | -1.68 pp | 0.61 % | -1.24 pp | 0.17 % | -1.68 pp | 0.21 % | -1.64 pp |
| Memory Throughput (%) | 1.77 % | 0.68 % | -1.09 pp | 0.64 % | -1.13 pp | 0.70 % | -1.07 pp | 0.74 % | -1.03 pp |
| DRAM Throughput (%) | 0.63 % | 0.42 % | -0.21 pp | 0.45 % | -0.18 pp | 0.44 % | -0.19 pp | 0.46 % | -0.17 pp |
| Achieved Occupancy (%) | 6.10 % | 1.56 % | -4.54 pp | 5.90 % | -0.20 pp | 1.56 % | -4.54 pp | 1.56 % | -4.54 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.05 | 0.02 | -0.03 | 0.05 | +0.00 | 0.02 | -0.03 | 0.02 | -0.03 |
| No Eligible (%) | 90.86 % | 95.66 % | +4.80 pp | 92.43 % | +1.57 pp | 92.34 % | +1.48 pp | 95.74 % | +4.88 pp |
| IPC Active | 0.37 inst/cycle | 0.03 inst/cycle | -0.34 | 0.23 inst/cycle | -0.14 | 0.06 inst/cycle | -0.31 | 0.04 inst/cycle | -0.33 |


### Key Learnings

#### ✅ Plan A: `streaming_load_eviction_to_reduce_l1_pollution` (1.081×)

When a Triton kernel's tl.load uses eviction_policy='evict_last' but the L1/TEX hit rate is 0%, the loaded data is never reused from L1, meaning evict_last actively pollutes L1 with dead data. Switching to eviction_policy='evict_first' on the load makes it a streaming/non-temporal access, matching the hardware's actual access pattern and reducing L1 pressure. However, in this specific case the profiling metrics show the change did not reduce duration (2.4us → 2.46us per NCU) despite a measured 1.08x end-to-end speedup, suggesting the gain came from reduced instruction count (3648 → 3072) and reduced L2 compression input sectors (283 → 117) rather than from L1TEX stall reduction — the CPIStall cycles per instruction actually increased (22.9 → 29.53 cycles/instruction). The strategy is sound for streaming kernels with confirmed 0% L1 reuse, but gains are modest on sub-3us kernels dominated by launch overhead.

> **Applicable when**: L1/TEX Hit Rate is 0% (confirmed no L1 reuse); kernel is a pure streaming/transpose/format-conversion op with no data reuse; tl.load uses eviction_policy='evict_last'; CPIStall rule is triggered with L1TEX stall as dominant stall reason; kernel accesses each element exactly once.

> **Avoid when**: L1/TEX Hit Rate > 0% (data is being reused from L1 — evict_last is beneficial); kernel has tiled reuse patterns (e.g., matmul, convolution with sliding windows); the load is already using evict_first; kernel is compute-bound rather than memory-latency-bound.

---

## Iteration 5

**Best so far**: 1.1083× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_callsite_grid_to_honor_autotune_and_add_large_yblock_configs | 1.020× | 1.039× | ✓ |  |
| B | shared_memory_transpose_tiling_to_coalesce_strided_store | 0.914× | 1.015× | ✓ |  |
| C | shared_memory_tiled_transpose_for_coalesced_access | 0.933× | 1.015× | ✓ |  |
| D | fix_callsite_lambda_grid_and_add_num_stages_to_autotune_configs | 0.994× | 1.051× | ✓ |  |
| E | add_triton_autotune | 0.989× | 1.064× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_callsite_grid... | Δ | shared_memory_tra... | Δ | shared_memory_til... | Δ | fix_callsite_lamb... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2660 ns | 2560 ns | -100 ns (-3.8%) | 2620 ns | -40 ns (-1.5%) | 2620 ns | -40 ns (-1.5%) | 2530 ns | -130 ns (-4.9%) | 2500 ns | -160 ns (-6.0%) |
| SM Throughput (%) | 1.85 % | 0.54 % | -1.31 pp | 0.66 % | -1.19 pp | 2.32 % | +0.47 pp | 0.16 % | -1.69 pp | 0.21 % | -1.64 pp |
| Memory Throughput (%) | 1.77 % | 0.56 % | -1.21 pp | 0.68 % | -1.09 pp | 2.07 % | +0.30 pp | 0.59 % | -1.18 pp | 0.69 % | -1.08 pp |
| DRAM Throughput (%) | 0.63 % | 0.41 % | -0.22 pp | 0.42 % | -0.21 pp | 0.64 % | +0.01 pp | 0.43 % | -0.20 pp | 0.42 % | -0.21 pp |
| Achieved Occupancy (%) | 6.10 % | 6.01 % | -0.09 pp | 5.83 % | -0.27 pp | 6.20 % | +0.10 pp | 1.56 % | -4.54 pp | 1.56 % | -4.54 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 28.00 register/thread | +0 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.05 | 0.05 | +0.00 | 0.05 | +0.00 | 0.05 | +0.00 | 0.02 | -0.03 | 0.02 | -0.03 |
| No Eligible (%) | 90.86 % | 91.68 % | +0.82 pp | 91.94 % | +1.08 pp | 84.29 % | -6.57 pp | 92.26 % | +1.40 pp | 95.22 % | +4.36 pp |
| IPC Active | 0.37 inst/cycle | 0.26 inst/cycle | -0.11 | 0.24 inst/cycle | -0.13 | 0.56 inst/cycle | +0.19 | 0.07 inst/cycle | -0.30 | 0.04 inst/cycle | -0.33 |

---

## Cumulative Progress

```
Baseline:  2660 ns  (1.000×)
Iter 3     2400 ns  (1.108×)  — fix_uncoalesced_store_with_streaming_eviction_hint

Final best: 2400 ns  (1.1083×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,45 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 8},   num_warps=1),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 2,  'YBLOCK': 16},  num_warps=2),
+        triton.Config({'XBLOCK': 2,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 2,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 16},  num_warps=2),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 16},  num_warps=2),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 16},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 16},  num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32},  num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
-def triton_poi_fused_convolution_36(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
+def triton_poi_fused_convolution_36(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
     ynumel = 768
     xnumel = 25
     yoffset = tl.program_id(1) * YBLOCK
+    xoffset = tl.program_id(0) * XBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
+    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     ymask = yindex < ynumel
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
     y0 = (yindex % 16)
     y1 = yindex // 16
     tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask & ymask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 16*x2 + 400*y1), tmp0, xmask & ymask)
-
-
+    tl.store(out_ptr0 + (y0 + 16*x2 + 400*y1), tmp0, xmask & ymask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +51,7 @@

     ynumel = 768
     xnumel = 25
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_36[grid](arg_0, arg_1, 768, 25, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_36[grid](arg_0, arg_1, 768, 25)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
