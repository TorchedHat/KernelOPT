# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3490 ns — **Best latency**: 3390 ns (**1.0295×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_tile_sizes_for_underutilized_transpose_kernel`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_28_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1408,1,1), Waves/SM: 0.67
- **Registers/thread**: 28
- **Duration**: 3490 ns
- **SM Throughput**: 21.41 % | **Memory**: 19.87 % | **DRAM**: 2.43 %
- **Achieved Occupancy**: 49.81 %
- **IPC Active**: 1.94 inst/cycle | **No Eligible**: 47.07 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_via_autotune_block_reshape | 0.986× | 1.039× | ✓ |  |
| B | autotune_tile_sizes_for_underutilized_transpose_kernel | 1.048× | 1.030× | ✓ | **✓ best** |
| C | fix_tile_sizes_and_add_autotune_for_transpose_kernel | 1.036× | 1.111× | ✓ |  |
| D | add_autotune_with_vectorized_loads_and_better_tiling | 1.047× | 1.171× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_tile_siz... | Δ | fix_tile_sizes_an... | Δ | add_autotune_with... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3490 ns | 3360 ns | -130 ns (-3.7%) | 3390 ns | -100 ns (-2.9%) | 3140 ns | -350 ns (-10.0%) | 2980 ns | -510 ns (-14.6%) |
| SM Throughput (%) | 21.41 % | 21.29 % | -0.12 pp | 18.99 % | -2.42 pp | 19.73 % | -1.68 pp | 8.83 % | -12.58 pp |
| Memory Throughput (%) | 19.87 % | 23.58 % | +3.71 pp | 9.03 % | -10.84 pp | 9.66 % | -10.21 pp | 7.70 % | -12.17 pp |
| DRAM Throughput (%) | 2.43 % | 2.52 % | +0.09 pp | 2.48 % | +0.05 pp | 2.70 % | +0.27 pp | 2.83 % | +0.40 pp |
| Achieved Occupancy (%) | 49.81 % | 25.51 % | -24.30 pp | 45.12 % | -4.69 pp | 49.19 % | -0.62 pp | 20.88 % | -28.93 pp |
| Registers/Thread | 28.00 register/thread | 29.00 register/thread | +1 | 28.00 register/thread | +0 | 22.00 register/thread | -6 | 21.00 register/thread | -7 |
| Waves/SM | 0.67 | 0.33 | -0.34 | 0.67 | +0.00 | 0.67 | +0.00 | 0.33 | -0.34 |
| No Eligible (%) | 47.07 % | 45.70 % | -1.37 pp | 47.43 % | +0.36 pp | 41.29 % | -5.78 pp | 71.07 % | +24.00 pp |
| IPC Active | 1.94 inst/cycle | 2.04 inst/cycle | +0.10 | 2.28 inst/cycle | +0.34 | 1.69 inst/cycle | -0.25 | 1.08 inst/cycle | -0.86 |

---

## Iteration 2

**Best so far**: 1.0295× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_1d_flat_grid | 0.015× | 0.007× | ✓ |  |
| B | persistent_tile_loop_scheduling_for_small_grid_transpose | 0.005× | 0.002× | ✓ |  |
| C | fix_hardcoded_grid_that_bypasses_autotune_and_add_num_stages | 1.010× | 1.020× | ✓ |  |
| D | fix_hardcoded_grid_to_enable_autotune_with_xblock16 | 0.992× | 1.081× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_l... | Δ | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3490 ns | 530340 ns | +526850 ns (+15096.0%) | 1470000 ns | +1466510 ns (+42020.3%) | 3420 ns | -70 ns (-2.0%) | 3230 ns | -260 ns (-7.4%) |
| SM Throughput (%) | 21.41 % | 30.47 % | +9.06 pp | 58.49 % | +37.08 pp | 21.32 % | -0.09 pp | 29.72 % | +8.31 pp |
| Memory Throughput (%) | 19.87 % | 54.57 % | +34.70 pp | 79.24 % | +59.37 pp | 23.59 % | +3.72 pp | 20.06 % | +0.19 pp |
| DRAM Throughput (%) | 2.43 % | 0.02 % | -2.41 pp | 0.01 % | -2.42 pp | 2.48 % | +0.05 pp | 2.60 % | +0.17 pp |
| Achieved Occupancy (%) | 49.81 % | 33.32 % | -16.49 pp | 61.86 % | +12.05 pp | 25.43 % | -24.38 pp | 46.89 % | -2.92 pp |
| Registers/Thread | 28.00 register/thread | 25.00 register/thread | -3 | 29.00 register/thread | +1 | 29.00 register/thread | +1 | 27.00 register/thread | -1 |
| Waves/SM | 0.67 | 0.33 | -0.34 | 0.67 | +0.00 | 0.33 | -0.34 | 0.67 | +0.00 |
| No Eligible (%) | 47.07 % | 70.20 % | +23.13 pp | 38.81 % | -8.26 pp | 45.92 % | -1.15 pp | 37.41 % | -9.66 pp |
| IPC Active | 1.94 inst/cycle | 1.19 inst/cycle | -0.75 | 2.41 inst/cycle | +0.47 | 2.04 inst/cycle | +0.10 | 2.54 inst/cycle | +0.60 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling_1d_flat_grid` (0.007×)

Converting a 2D fractional-wave Triton kernel to a persistent 1D grid with a while-loop caused a 150x slowdown (2.98 µs → 530 µs). The while-loop body executed ~596x more instructions (209K → 125M) because the XBLOCK/TILES_PER_SM autotune configs were misapplied: XBLOCK was repurposed as the flat tile size and XBLOCK values like 128–2048 were used, but the autotuner selected a config that caused each CTA to iterate over many tiny tiles in a tight loop rather than covering the work in one pass. The persistent grid approach adds significant per-iteration overhead (integer div/mod for index recovery, branch instructions, loop control) that dominates when the total work is tiny (~101K elements, ~3 µs baseline). Achieved occupancy barely improved (20.9% → 33.3%) while eligible warps per scheduler actually dropped (0.72 → 0.47), and a new CPIStall on L1TEX appeared (64.6% of stall cycles) due to uncoalesced global stores introduced by the scatter-style write pattern in the flat loop.

> **Applicable when**: NEVER: persistent 1D grid with while-loop is harmful for kernels with total work < ~1M elements and baseline duration < ~10 µs; the loop overhead and index recovery cost exceed any occupancy benefit

> **Avoid when**: Kernel has fewer than ~1M total elements or baseline duration under ~10 µs; the work per CTA in the while-loop is less than ~1000 elements; the output write pattern is a scatter (non-sequential index formula) that becomes uncoalesced when flattened to 1D; the fractional-wave problem is caused by small grid size rather than insufficient parallelism in the algorithm itself

#### ❌ Plan B: `persistent_tile_loop_scheduling_for_small_grid_transpose` (0.002×)

Replacing a standard multi-CTA grid with a persistent kernel (one CTA per SM iterating over tiles in a while-loop) caused a ~500x increase in elapsed cycles (4411 → 2,204,310) and a ~3250x increase in executed instructions (209,792 → 682,192,896). The root cause is an implementation flaw: the kernel launch site was NOT updated to use the new persistent grid (132 CTAs); instead the old grid formula `(cdiv(9,128), cdiv(11264,8)) = (1, 1408)` was kept, so 1408 CTAs each ran the full persistent while-loop over all 176 Y-tiles instead of striding by num_sms. This caused each element to be processed ~1408 times instead of once, turning a microsecond kernel into a millisecond one. The persistent kernel pattern itself is not inherently wrong, but the call site grid must be changed to `(NUM_SMS, 1)` atomically with the kernel body change.

> **Applicable when**: Never apply this specific broken pattern. The persistent kernel strategy itself is valid when: grid is correctly set to (NUM_SMS, 1), YBLOCK is tuned to be larger than the original, and the kernel body uses tl.num_programs(0) for the stride.

> **Avoid when**: Do not apply persistent kernel refactoring when the call site grid formula cannot be atomically updated alongside the kernel body. Do not apply when the kernel is already very short-lived (< 5 us) and the persistent loop overhead would dominate. Do not apply when the original grid already achieves >= 1 wave per SM.

---

## Iteration 3

**Best so far**: 1.0295× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_use_autotune_meta_tile_sizes | 1.025× | 1.058× | ✓ |  |
| B | fix_call_site_grid_to_enable_autotune_lambda_for_transpose_kernel | 1.009× | 1.020× | ✓ |  |
| C | revert_persistent_loop_to_2d_grid_with_rightsized_xblock | 1.039× | 1.148× | ✓ |  |
| D | revert_broken_persistent_loop_and_autotune_2d_grid_with_right_sized_xblock | 1.046× | 1.137× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_call_site_gri... | Δ | revert_persistent... | Δ | revert_broken_per... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3490 ns | 3300 ns | -190 ns (-5.4%) | 3420 ns | -70 ns (-2.0%) | 3040 ns | -450 ns (-12.9%) | 3070 ns | -420 ns (-12.0%) |
| SM Throughput (%) | 21.41 % | 27.65 % | +6.24 pp | 21.33 % | -0.08 pp | 14.29 % | -7.12 pp | 14.61 % | -6.80 pp |
| Memory Throughput (%) | 19.87 % | 18.64 % | -1.23 pp | 23.59 % | +3.72 pp | 12.51 % | -7.36 pp | 12.73 % | -7.14 pp |
| DRAM Throughput (%) | 2.43 % | 2.56 % | +0.13 pp | 2.47 % | +0.04 pp | 2.76 % | +0.33 pp | 2.75 % | +0.32 pp |
| Achieved Occupancy (%) | 49.81 % | 47.04 % | -2.77 pp | 24.73 % | -25.08 pp | 26.31 % | -23.50 pp | 23.22 % | -26.59 pp |
| Registers/Thread | 28.00 register/thread | 27.00 register/thread | -1 | 29.00 register/thread | +1 | 30.00 register/thread | +2 | 30.00 register/thread | +2 |
| Waves/SM | 0.67 | 0.67 | +0.00 | 0.33 | -0.34 | 0.33 | -0.34 | 0.33 | -0.34 |
| No Eligible (%) | 47.07 % | 34.55 % | -12.52 pp | 47.08 % | +0.01 pp | 59.09 % | +12.02 pp | 50.80 % | +3.73 pp |
| IPC Active | 1.94 inst/cycle | 2.52 inst/cycle | +0.58 | 1.97 inst/cycle | +0.03 | 1.54 inst/cycle | -0.40 | 1.74 inst/cycle | -0.20 |


### Key Learnings

#### ✅ Plan A: `fix_hardcoded_grid_to_use_autotune_meta_tile_sizes` (1.058×)

When a Triton kernel's call-site grid uses hardcoded denominators (e.g., `cdiv(ynumel, 8)`) instead of `META['YBLOCK']`, the autotuner's tile-size selection is completely decoupled from the actual grid dimensions, causing the kernel to launch with a grid sized for the smallest tile regardless of which config is chosen. Replacing the hardcoded grid with a `lambda META: (cdiv(xnumel, META['XBLOCK']), cdiv(ynumel, META['YBLOCK']))` allows the autotuner to correctly pair tile sizes with grid dimensions. In this case the fix yielded only a modest 5.8% speedup rather than the predicted 10–50%, because the profiling metrics (waves/SM, achieved occupancy, No Eligible %) barely changed, suggesting the autotuner still selected a config close to the original YBLOCK=8 or the workload is too small for larger tiles to help.

> **Applicable when**: Triton kernel has @triton.autotune with multiple XBLOCK/YBLOCK configs but the call-site grid uses hardcoded integer denominators instead of META['XBLOCK']/META['YBLOCK']; waves_per_sm < 1.0 and AchievedOccupancy << TheoreticalOccupancy; SOLBottleneck rule triggered.

> **Avoid when**: The kernel is already using a META lambda grid; the workload is so small (total elements < ~10K) that even the smallest autotune tile config covers all work in one wave; the autotuner's best config happens to match the hardcoded denominator (no effective change); when the speedup ceiling is low due to other bottlenecks (e.g., L2 workload imbalance, uncoalesced access patterns) dominating.

---

## Cumulative Progress

```
Baseline:  3490 ns  (1.000×)
Iter 1     3390 ns  (1.030×)  — autotune_tile_sizes_for_underutilized_transpose_kernel

Final best: 3390 ns  (1.0295×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,25 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 512}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 11264
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = (yindex < ynumel)
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +31,8 @@

     y3 = yindex
     y0 = (yindex % 352)
     y1 = yindex // 352
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 352*x2 + 3168*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 352*x2 + 3168*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +40,7 @@

     ynumel = 11264
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28[grid](arg_0, arg_1, 11264, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28[grid](arg_0, arg_1, 11264, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
