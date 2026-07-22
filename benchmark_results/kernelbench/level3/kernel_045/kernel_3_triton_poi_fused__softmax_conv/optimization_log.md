# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__softmax_convolution_exp_sub_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2620 ns — **Best latency**: 2460 ns (**1.0650×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`fix_tile_sizes_for_small_xnumel_transpose_kernel`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__softmax_convolution_exp_sub_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_3_triton_poi_fused__softmax_conv/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 0.24
- **Registers/thread**: 30
- **Duration**: 2620 ns
- **SM Throughput**: 7.44 % | **Memory**: 8.83 % | **DRAM**: 1.18 %
- **Achieved Occupancy**: 20.86 %
- **IPC Active**: 0.93 inst/cycle | **No Eligible**: 76.33 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_autotune | — | 0.963× | ✓ |  |
| B | autotune_tile_sizes_for_underutilized_kernel | — | fail | ✓ |  |
| C | right_size_tile_blocks_for_small_xnumel | — | 0.953× | ✓ |  |
| D | fix_tile_sizes_for_small_xnumel_transpose_kernel | — | 1.065× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | right_size_tile_b... | Δ | fix_tile_sizes_fo... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2620 ns | 2720 ns | +100 ns (+3.8%) | 2750 ns | +130 ns (+5.0%) | 2460 ns | -160 ns (-6.1%) |
| SM Throughput (%) | 7.44 % | 4.56 % | -2.88 pp | 2.98 % | -4.46 pp | 4.43 % | -3.01 pp |
| Memory Throughput (%) | 8.83 % | 7.70 % | -1.13 pp | 3.00 % | -5.83 pp | 4.97 % | -3.86 pp |
| DRAM Throughput (%) | 1.18 % | 4.48 % | +3.30 pp | 1.14 % | -0.04 pp | 1.27 % | +0.09 pp |
| Achieved Occupancy (%) | 20.86 % | 10.82 % | -10.04 pp | 8.34 % | -12.52 pp | 9.00 % | -11.86 pp |
| Registers/Thread | 30.00 register/thread | 32.00 register/thread | +2 | 21.00 register/thread | -9 | 32.00 register/thread | +2 |
| Waves/SM | 0.24 | 0.12 | -0.12 | 0.12 | -0.12 | 0.12 | -0.12 |
| No Eligible (%) | 76.33 % | 85.55 % | +9.22 pp | 89.87 % | +13.54 pp | 84.30 % | +7.97 pp |
| IPC Active | 0.93 inst/cycle | 0.54 inst/cycle | -0.39 | 0.39 inst/cycle | -0.54 | 0.50 inst/cycle | -0.43 |

---

## Iteration 2

**Best so far**: 1.065× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_loop_with_flattened_1d_grid | — | 0.036× | ✓ |  |
| B | fix_autotune_grid_lambda_and_expand_yblock_configs | — | 0.963× | ✓ |  |
| C | persistent_tile_loop_for_underutilized_transpose_kernel | — | 0.010× | ✓ |  |
| D | add_triton_autotune | — | 0.963× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_l... | Δ | fix_autotune_grid... | Δ | persistent_tile_l... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2620 ns | 72610 ns | +69990 ns (+2671.4%) | 2720 ns | +100 ns (+3.8%) | 255420 ns | +252800 ns (+9648.9%) | 2720 ns | +100 ns (+3.8%) |
| SM Throughput (%) | 7.44 % | 14.90 % | +7.46 pp | 1.45 % | -5.99 pp | 35.77 % | +28.33 pp | 4.73 % | -2.71 pp |
| Memory Throughput (%) | 8.83 % | 27.80 % | +18.97 pp | 4.36 % | -4.47 pp | 59.51 % | +50.68 pp | 5.33 % | -3.50 pp |
| DRAM Throughput (%) | 1.18 % | 0.04 % | -1.14 pp | 1.15 % | -0.03 pp | 0.01 % | -1.17 pp | 1.15 % | -0.03 pp |
| Achieved Occupancy (%) | 20.86 % | 12.11 % | -8.75 pp | 5.27 % | -15.59 pp | 24.24 % | +3.38 pp | 8.50 % | -12.36 pp |
| Registers/Thread | 30.00 register/thread | 31.00 register/thread | +1 | 20.00 register/thread | -10 | 32.00 register/thread | +2 | 32.00 register/thread | +2 |
| Waves/SM | 0.24 | 0.12 | -0.12 | 0.12 | -0.12 | 0.24 | +0.00 | 0.12 | -0.12 |
| No Eligible (%) | 76.33 % | 84.31 % | +7.98 pp | 93.09 % | +16.76 pp | 63.61 % | -12.72 pp | 80.36 % | +4.03 pp |
| IPC Active | 0.93 inst/cycle | 0.63 inst/cycle | -0.30 | 0.17 inst/cycle | -0.76 | 1.44 inst/cycle | +0.51 | 0.67 inst/cycle | -0.26 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_loop_with_flattened_1d_grid` (0.035×)

Replacing a direct 2D grid dispatch with a persistent `tl.range(cta_id, num_tiles, num_ctas)` loop inside a Triton kernel caused a ~28x slowdown (2.72us → 74.21us) on a tiny 4096×9 transpose. Executed instructions exploded from 32,768 to 8,694,784 (~265x), because each of the 512 CTAs iterated over many tiles with scalar modulo/division overhead (`xtile = tile_id % num_xtiles`, `ytile = tile_id // num_xtiles`) dominating execution. Waves per SM remained 0.12 and memory throughput collapsed from 55.53 GB/s to 2.05 GB/s, confirming the loop overhead serialized work rather than improving parallelism. This is the same anti-pattern as exp_002 — the persistent loop is fundamentally unsuitable for small problems where total_tiles is comparable to the CTA count.

> **Applicable when**: Never beneficial for tiny kernels (<200 KB data, <10us duration) where loop overhead dominates. Persistent loops are only beneficial when total_tiles >> num_SMs and per-tile work is substantial enough to amortize loop index arithmetic.

> **Avoid when**: Total problem size is small (e.g., <200 KB), kernel duration is already in the low-microsecond range (<10us), or the grid is already fixed at a size that cannot be reduced by the persistent loop (waves_per_sm stays the same). Avoid when tile index arithmetic (modulo/division) inside the loop body would dominate execution time relative to the actual memory work per tile.

#### ❌ Plan C: `persistent_tile_loop_for_underutilized_transpose_kernel` (0.010×)

Replacing a simple 2D grid dispatch with a persistent while-loop inside the Triton kernel caused a ~104x regression (2.46us → 255.42us). The root cause was a critical implementation flaw: the grid was left as the original broken formula `(cdiv(xnumel, 128), cdiv(ynumel, 8)) = (1, 512)` while the kernel was rewritten to use `XBLOCK=128, YBLOCK=8` as constexpr arguments — this caused each of the 512 CTAs to loop over all 128 tiles (instead of 1 tile each), executing 72M instructions vs. the original 97K (a 742x instruction increase). The CPIStall rule appeared (37% of cycles at barriers) and executed instructions exploded from 97,536 to 72,411,136, confirming the tile loop ran far more iterations than intended due to the mismatched grid/tile-count relationship.

> **Applicable when**: Never apply this specific broken pattern. The persistent tile loop strategy itself is valid when: grid size is correctly set to min(SM_count, total_tiles) as a 1D grid, XBLOCK/YBLOCK constexpr values match the tile-count computation, and the while-loop stride equals the actual grid size.

> **Avoid when**: Do not apply a persistent while-loop inside a Triton kernel when the grid formula is not simultaneously updated to a 1D grid of size min(SM_count, total_tiles). Mixing a 2D legacy grid with a 1D persistent loop body causes each CTA to redundantly process all tiles, exploding instruction count by orders of magnitude. Also avoid when the kernel is already very short (<5us) and the problem is simply too small to fill the GPU — the overhead of the loop itself may dominate.

---

## Iteration 3

**Best so far**: 1.065× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_match_autotuned_tile_sizes | — | fail | ✓ |  |
| B | fix_hardcoded_grid_to_use_autotune_meta_lambda | — | 0.035× | ✓ |  |
| C | revert_broken_persistent_loop_to_clean_2d_grid_with_right_sized_xblock | — | 1.048× | ✓ |  |
| D | revert_broken_persistent_loop_to_clean_2d_grid | — | 1.048× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | revert_broken_per... | Δ | revert_broken_per... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2620 ns | 74210 ns | +71590 ns (+2732.4%) | 2500 ns | -120 ns (-4.6%) | 2500 ns | -120 ns (-4.6%) |
| SM Throughput (%) | 7.44 % | 14.73 % | +7.29 pp | 5.42 % | -2.02 pp | 4.81 % | -2.63 pp |
| Memory Throughput (%) | 8.83 % | 27.48 % | +18.65 pp | 5.65 % | -3.18 pp | 5.42 % | -3.41 pp |
| DRAM Throughput (%) | 1.18 % | 0.04 % | -1.14 pp | 1.26 % | +0.08 pp | 1.25 % | +0.07 pp |
| Achieved Occupancy (%) | 20.86 % | 12.11 % | -8.75 pp | 8.18 % | -12.68 pp | 8.38 % | -12.48 pp |
| Registers/Thread | 30.00 register/thread | 31.00 register/thread | +1 | 32.00 register/thread | +2 | 32.00 register/thread | +2 |
| Waves/SM | 0.24 | 0.12 | -0.12 | 0.12 | -0.12 | 0.12 | -0.12 |
| No Eligible (%) | 76.33 % | 83.80 % | +7.47 pp | 78.01 % | +1.68 pp | 78.29 % | +1.96 pp |
| IPC Active | 0.93 inst/cycle | 0.62 inst/cycle | -0.31 | 0.64 inst/cycle | -0.29 | 0.64 inst/cycle | -0.29 |

---

## Cumulative Progress

```
Baseline:  2620 ns  (1.000×)
Iter 1     2460 ns  (1.065×)  — fix_tile_sizes_for_small_xnumel_transpose_kernel

Final best: 2460 ns  (1.0650×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  32}, num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  64}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 32, 'YBLOCK':  32}, num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK':  64}, num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__softmax_convolution_exp_sub_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4096
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,9 +32,8 @@

     y3 = yindex
     y0 = (yindex % 64)
     y1 = yindex // 64
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), mask=xmask)
-    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, mask=xmask)
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), mask=xmask & ymask)
+    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -29,7 +41,7 @@

     ynumel = 4096
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__softmax_convolution_exp_sub_3[grid](arg_0, arg_1, 4096, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__softmax_convolution_exp_sub_3[grid](arg_0, arg_1, 4096, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
