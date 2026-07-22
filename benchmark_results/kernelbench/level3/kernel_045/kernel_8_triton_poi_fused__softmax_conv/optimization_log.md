# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__softmax_convolution_exp_sub_8` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3870 ns — **Best latency**: 3810 ns (**1.0157×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`optimize_tile_sizes_for_small_xnumel_kernel`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__softmax_convolution_exp_sub_8`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_8_triton_poi_fused__softmax_conv/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2048,1,1), Waves/SM: 0.97
- **Registers/thread**: 28
- **Duration**: 3870 ns
- **SM Throughput**: 25.95 % | **Memory**: 24.83 % | **DRAM**: 3.16 %
- **Achieved Occupancy**: 65.64 %
- **IPC Active**: 2.28 inst/cycle | **No Eligible**: 40.55 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | optimize_tile_sizes_for_small_xnumel_kernel | — | 1.016× | ✓ | **✓ best** |
| C | fix_tile_sizes_and_autotune_for_underutilized_transpose_kernel | — | 1.173× | ✓ |  |
| D | fix_tile_dimensions_and_add_autotune_for_coalesced_vectorized_access | — | 0.780× | ✓ |  |


### Metric Comparison

| Metric | Baseline | optimize_tile_siz... | Δ | fix_tile_sizes_an... | Δ | fix_tile_dimensio... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3870 ns | 3810 ns | -60 ns (-1.6%) | 3300 ns | -570 ns (-14.7%) | 4960 ns | +1090 ns (+28.2%) |
| SM Throughput (%) | 25.95 % | 37.59 % | +11.64 pp | 15.24 % | -10.71 pp | 49.10 % | +23.15 pp |
| Memory Throughput (%) | 24.83 % | 25.59 % | +0.76 pp | 6.93 % | -17.90 pp | 41.58 % | +16.75 pp |
| DRAM Throughput (%) | 3.16 % | 3.20 % | +0.04 pp | 3.68 % | +0.52 pp | 2.47 % | -0.69 pp |
| Achieved Occupancy (%) | 65.64 % | 61.91 % | -3.73 pp | 57.61 % | -8.03 pp | 63.10 % | -2.54 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 18.00 register/thread | -10 | 32.00 register/thread | +4 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 40.55 % | 30.57 % | -9.98 pp | 41.89 % | +1.34 pp | 22.64 % | -17.91 pp |
| IPC Active | 2.28 inst/cycle | 2.78 inst/cycle | +0.50 | 2.04 inst/cycle | -0.24 | 3.12 inst/cycle | +0.84 |

---

## Iteration 2

**Best so far**: 1.0157× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_via_grid_lambda_fix | — | 1.142× | ✓ |  |
| B | persistent_tile_scheduling | — | 1.173× | ✓ |  |
| C | persistent_tile_scheduling | — | 1.187× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3870 ns | 3390 ns | -480 ns (-12.4%) | 3300 ns | -570 ns (-14.7%) | 3260 ns | -610 ns (-15.8%) |
| SM Throughput (%) | 25.95 % | 15.08 % | -10.87 pp | 15.33 % | -10.62 pp | 13.12 % | -12.83 pp |
| Memory Throughput (%) | 24.83 % | 6.86 % | -17.97 pp | 6.98 % | -17.85 pp | 3.92 % | -20.91 pp |
| DRAM Throughput (%) | 3.16 % | 3.57 % | +0.41 pp | 3.67 % | +0.51 pp | 3.76 % | +0.60 pp |
| Achieved Occupancy (%) | 65.64 % | 57.98 % | -7.66 pp | 57.56 % | -8.08 pp | 55.64 % | -10.00 pp |
| Registers/Thread | 28.00 register/thread | 18.00 register/thread | -10 | 18.00 register/thread | -10 | 16.00 register/thread | -12 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 40.55 % | 50.37 % | +9.82 pp | 42.49 % | +1.94 pp | 44.99 % | +4.44 pp |
| IPC Active | 2.28 inst/cycle | 2.04 inst/cycle | -0.24 | 2.11 inst/cycle | -0.17 | 1.93 inst/cycle | -0.35 |

---

## Iteration 3

**Best so far**: 1.0157× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_match_persistent_tile_loop_autotune | — | 1.187× | ✓ |  |
| B | persistent_ytile_loop_with_lambda_grid_fix | — | 1.142× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | persistent_ytile_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3870 ns | 3260 ns | -610 ns (-15.8%) | 3390 ns | -480 ns (-12.4%) |
| SM Throughput (%) | 25.95 % | 12.68 % | -13.27 pp | 28.39 % | +2.44 pp |
| Memory Throughput (%) | 24.83 % | 3.78 % | -21.05 pp | 12.59 % | -12.24 pp |
| DRAM Throughput (%) | 3.16 % | 3.75 % | +0.59 pp | 3.58 % | +0.42 pp |
| Achieved Occupancy (%) | 65.64 % | 53.70 % | -11.94 pp | 59.82 % | -5.82 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 24.00 register/thread | -4 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 40.55 % | 58.33 % | +17.78 pp | 28.40 % | -12.15 pp |
| IPC Active | 2.28 inst/cycle | 1.88 inst/cycle | -0.40 | 2.76 inst/cycle | +0.48 |


### Key Learnings

#### ✅ Plan A: `fix_grid_lambda_to_match_persistent_tile_loop_autotune` (1.187×)

Fixing the Triton launch grid from a static tuple to a lambda that reads META['XBLOCK'] and META['YBLOCK'] while also dividing the Y dimension by TILES_PER_CTA=2 (to match the persistent loop in the kernel body) produced a 1.19x wall-clock speedup, but NCU per-kernel metrics were essentially unchanged or slightly worse: Duration stayed at 3.26 µs, Elapsed Cycles dropped only from 4829 to 4823, and No Eligible warps actually worsened from 45% to 58%. This indicates the wall-clock speedup comes from autotuner-level effects (correct config selection across benchmark runs) rather than per-launch SM utilization improvement. The persistent loop + grid lambda combination is the correct structural pairing — a persistent loop without the matching halved grid Y dimension causes double-coverage of output tiles, and the grid lambda without the /2 divisor causes undercoverage — but the NCU-visible benefit is minimal once instruction count and register pressure are already optimized.

> **Applicable when**: Triton kernel with a persistent Y-tile loop (fixed trip count TILES_PER_CTA) where the launch grid is hardcoded as a static tuple that neither reads META tile sizes nor divides by TILES_PER_CTA. Both mismatches must be fixed together: the grid must be a lambda AND divide Y by TILES_PER_CTA. Symptoms: SOLBottleneck with ~1 wave, IssueSlotUtilization firing, static grid constants that don't match any autotune config.

> **Avoid when**: When the kernel does NOT have a persistent tile loop (TILES_PER_CTA=1), use the plain grid lambda without the /TILES_PER_CTA divisor. Also avoid if NCU per-kernel metrics are the primary optimization target — this fix primarily improves end-to-end benchmark timing via correct autotuner config selection, not per-launch SM utilization. Do not apply if the grid Y dimension is already a lambda reading META values.

---

## Cumulative Progress

```
Baseline:  3870 ns  (1.000×)
Iter 1     3810 ns  (1.016×)  — optimize_tile_sizes_for_small_xnumel_kernel

Final best: 3810 ns  (1.0157×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,23 +5,41 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 64},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 32},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 512}, num_warps=8, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__softmax_convolution_exp_sub_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 16384
     xnumel = 9
-    yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
-    y3 = yindex
-    y0 = (yindex % 128)
-    y1 = yindex // 128
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask & ymask)
-
+    pid_y = tl.program_id(1)
+    num_y_programs = tl.num_programs(1)
+    # Persistent tile loop: each CTA processes multiple Y-tiles
+    for tile_idx in range(2):
+        yoffset = (pid_y + tile_idx * num_y_programs) * YBLOCK
+        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
+        ymask = yindex < ynumel
+        y3 = yindex
+        y0 = (yindex % 128)
+        y1 = yindex // 128
+        tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, other=0.0, eviction_policy='evict_last')
+        tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -29,7 +47,7 @@

     ynumel = 16384
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__softmax_convolution_exp_sub_8[grid](arg_0, arg_1, 16384, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__softmax_convolution_exp_sub_8[grid](arg_0, arg_1, 16384, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
