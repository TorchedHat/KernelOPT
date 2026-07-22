# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2850 ns — **Best latency**: 2660 ns (**1.0714×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`add_triton_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_6`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_6_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (384,1,1), Waves/SM: 0.18
- **Registers/thread**: 28
- **Duration**: 2850 ns
- **SM Throughput**: 7.60 % | **Memory**: 6.92 % | **DRAM**: 0.83 %
- **Achieved Occupancy**: 16.36 %
- **IPC Active**: 0.89 inst/cycle | **No Eligible**: 75.08 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_small_grid_utilization | 1.006× | 0.848× | ✓ |  |
| C | autotune_tile_sizes_for_underutilized_small_kernel | — | fail | ✓ |  |
| D | autotune_tile_sizes_fix_grid_underutilization | 0.917× | 0.864× | ✓ |  |
| E | add_triton_autotune | 1.012× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2850 ns | 3360 ns | +510 ns (+17.9%) | 3300 ns | +450 ns (+15.8%) | 2850 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 7.60 % | 2.23 % | -5.37 pp | 1.84 % | -5.76 pp | 6.73 % | -0.87 pp |
| Memory Throughput (%) | 6.92 % | 21.65 % | +14.73 pp | 22.07 % | +15.15 pp | 6.13 % | -0.79 pp |
| DRAM Throughput (%) | 0.83 % | 21.65 % | +20.82 pp | 22.07 % | +21.24 pp | 0.83 % | +0.00 pp |
| Achieved Occupancy (%) | 16.36 % | 14.33 % | -2.03 pp | 14.45 % | -1.91 pp | 15.99 % | -0.37 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 28.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.18 | +0.00 | 0.18 | +0.00 | 0.18 | +0.00 |
| No Eligible (%) | 75.08 % | 94.25 % | +19.17 pp | 95.07 % | +19.99 pp | 76.38 % | +1.30 pp |
| IPC Active | 0.89 inst/cycle | 0.18 inst/cycle | -0.71 | 0.15 inst/cycle | -0.74 | 0.87 inst/cycle | -0.02 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | persistent_tile_scheduling | 0.085× | 0.039× | ✓ |  |
| D | add_triton_autotune | 1.003× | 1.071× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2850 ns | 73120 ns | +70270 ns (+2465.6%) | 2660 ns | -190 ns (-6.7%) |
| SM Throughput (%) | 7.60 % | 15.08 % | +7.48 pp | 7.59 % | -0.01 pp |
| Memory Throughput (%) | 6.92 % | 29.81 % | +22.89 pp | 6.91 % | -0.01 pp |
| DRAM Throughput (%) | 0.83 % | 0.05 % | -0.78 pp | 0.89 % | +0.06 pp |
| Achieved Occupancy (%) | 16.36 % | 18.20 % | +1.84 pp | 16.07 % | -0.29 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 28.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.18 | +0.00 | 0.18 | +0.00 |
| No Eligible (%) | 75.08 % | 85.18 % | +10.10 pp | 77.67 % | +2.59 pp |
| IPC Active | 0.89 inst/cycle | 0.63 inst/cycle | -0.26 | 0.82 inst/cycle | -0.07 |


### Key Learnings

#### ❌ Plan B: `persistent_tile_scheduling` (0.038×)

Converting a 2D tiled kernel body to a persistent while-loop using `tl.program_id(0)` and `tl.num_programs(0)` while leaving the launch grid as a 2D tuple `(1, 384)` causes catastrophic regression (~26x slowdown). With a 2D grid, `tl.num_programs(0)` returns 1 (the axis-0 count), so the loop stride is `1 * BLOCK` and all 384 CTAs each iterate over the entire 27648-element work range, inflating executed instructions by ~56x (132K → 7.38M) and elapsed cycles by ~28x (4005 → 110934). Despite L1/TEX hit rate improving from 28% to 84% and L2 hit rate reaching 99.6% (data fits in cache after first pass), the redundant work overwhelms any cache benefit. This is the third consecutive iteration reproducing the same bug: persistent loop body + unchanged 2D grid = O(num_ctas) instruction inflation.

> **Applicable when**: Never — this is a pure implementation flaw reproduced three times. The persistent kernel pattern itself is valid but requires the grid to be simultaneously updated to a 1D tuple `(num_ctas,)` matching the axis used by `tl.program_id(0)` and `tl.num_programs(0)`.

> **Avoid when**: Never convert a kernel body to a persistent while-loop without simultaneously changing the launch grid from the original 2D tuple to a 1D tuple `(num_ctas,)`. If the grid remains 2D (e.g., `(1, 384)`), `tl.num_programs(0)` returns the axis-0 count (1, not 384), causing each CTA to stride by only `1 * BLOCK` and redundantly cover the entire work range. Instruction count scales linearly with the number of CTAs instead of being constant, producing O(num_ctas) regression. Adding @triton.autotune on top of this broken pattern does not fix the grid mismatch and may further obscure the root cause.

---

## Iteration 3

**Best so far**: 1.0714× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_yblock_to_reduce_barrier_stalls_and_improve_warp_utilization | 0.980× | 1.011× | ✓ |  |
| C | fix_broken_persistent_kernel_revert_to_2d_tiled | 0.931× | 1.048× | ✓ |  |
| D | fix_persistent_kernel_grid_to_eliminate_redundant_work | 0.085× | 0.040× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_yblock_t... | Δ | fix_broken_persis... | Δ | fix_persistent_ke... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2850 ns | 2820 ns | -30 ns (-1.1%) | 2720 ns | -130 ns (-4.6%) | 71040 ns | +68190 ns (+2392.6%) |
| SM Throughput (%) | 7.60 % | 7.47 % | -0.13 pp | 6.20 % | -1.40 pp | 14.58 % | +6.98 pp |
| Memory Throughput (%) | 6.92 % | 6.80 % | -0.12 pp | 6.70 % | -0.22 pp | 28.84 % | +21.92 pp |
| DRAM Throughput (%) | 0.83 % | 0.84 % | +0.01 pp | 0.88 % | +0.05 pp | 0.05 % | -0.78 pp |
| Achieved Occupancy (%) | 16.36 % | 15.94 % | -0.42 pp | 16.16 % | -0.20 pp | 18.18 % | +1.82 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 32.00 register/thread | +4 | 32.00 register/thread | +4 |
| Waves/SM | 0.18 | 0.18 | +0.00 | 0.18 | +0.00 | 0.18 | +0.00 |
| No Eligible (%) | 75.08 % | 80.63 % | +5.55 pp | 78.03 % | +2.95 pp | 85.07 % | +9.99 pp |
| IPC Active | 0.89 inst/cycle | 0.86 inst/cycle | -0.03 | 0.78 inst/cycle | -0.11 | 0.59 inst/cycle | -0.30 |


### Key Learnings

#### ✅ Plan C: `fix_broken_persistent_kernel_revert_to_2d_tiled` (1.048×)

When a persistent while-loop kernel is launched with a grid that does not match the loop stride (e.g., grid axis 0 = 1 but tl.num_programs(0) = 1 causes each CTA to iterate over all tiles instead of one), reverting to a simple 2D tiled kernel eliminates the redundant work. The revert reduced executed instructions from 158K to 132K and registers per thread from 28 to 32, yielding a modest 4.8% speedup; the instruction reduction came from removing the loop overhead and barrier stalls (CPIStall was 35% of stall cycles before). The key causal signal is a dramatic instruction count explosion relative to the expected element count — if executed instructions >> (grid_size × elements_per_block), a persistent loop with wrong stride is the likely culprit.

> **Applicable when**: A kernel's executed instruction count is much larger than expected (grid_size × BLOCK × ops_per_element), CPIStall is flagged as a top stall reason (>30% of stall cycles), and the kernel contains a while-loop driven by tl.num_programs(). This pattern indicates a persistent loop with a mismatched stride where each CTA processes far more tiles than intended.

> **Avoid when**: Do not revert a persistent kernel if the grid was correctly sized to num_sms and tl.num_programs() returns the correct stride — in that case the persistent loop is working as intended and removing it would reduce SM utilization. Also, this revert does not address the underlying SOLBottleneck (waves_per_sm remains 0.18); expect only small gains from removing loop overhead.

---

## Iteration 4

**Best so far**: 1.0714× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_yblock_with_correct_lambda_grid | 0.932× | 1.100× | ✓ |  |
| C | revert_broken_persistent_loop_to_clean_2d_tiled_kernel | 1.000× | 0.956× | ✓ |  |
| D | add_triton_autotune | 0.085× | 0.038× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_yblock_w... | Δ | revert_broken_per... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2850 ns | 2590 ns | -260 ns (-9.1%) | 2980 ns | +130 ns (+4.6%) | 73980 ns | +71130 ns (+2495.8%) |
| SM Throughput (%) | 7.60 % | 6.52 % | -1.08 pp | 7.84 % | +0.24 pp | 14.56 % | +6.96 pp |
| Memory Throughput (%) | 6.92 % | 7.05 % | +0.13 pp | 6.88 % | -0.04 pp | 28.75 % | +21.83 pp |
| DRAM Throughput (%) | 0.83 % | 0.91 % | +0.08 pp | 0.79 % | -0.04 pp | 0.05 % | -0.78 pp |
| Achieved Occupancy (%) | 16.36 % | 15.53 % | -0.83 pp | 16.18 % | -0.18 pp | 18.20 % | +1.84 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 30.00 register/thread | +2 | 32.00 register/thread | +4 |
| Waves/SM | 0.18 | 0.18 | +0.00 | 0.18 | +0.00 | 0.18 | +0.00 |
| No Eligible (%) | 75.08 % | 78.04 % | +2.96 pp | 77.00 % | +1.92 pp | 85.41 % | +10.33 pp |
| IPC Active | 0.89 inst/cycle | 0.73 inst/cycle | -0.16 | 0.75 inst/cycle | -0.14 | 0.63 inst/cycle | -0.26 |

---

## Cumulative Progress

```
Baseline:  2850 ns  (1.000×)
Iter 2     2660 ns  (1.071×)  — add_triton_autotune

Final best: 2660 ns  (1.0714×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,21 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
-    ynumel = 3072
-    xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
-    y0 = (yindex % 96)
-    y1 = yindex // 96
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 96*x2 + 864*y1), tmp0, xmask)
-
-
+    y0 = y3 % 96
+    y1 = y3 // 96
+    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), mask=xmask & ymask, other=0.0)
+    tl.store(out_ptr0 + (y0 + 96 * x2 + 864 * y1), tmp0, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
