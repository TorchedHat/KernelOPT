# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2660 ns — **Best latency**: 2750 ns (**0.9673×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_008/kernel_3_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 0.24
- **Registers/thread**: 28
- **Duration**: 2660 ns
- **SM Throughput**: 9.19 % | **Memory**: 8.68 % | **DRAM**: 1.17 %
- **Achieved Occupancy**: 21.19 %
- **IPC Active**: 1.13 inst/cycle | **No Eligible**: 75.47 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.027× | 0.010× | ✓ |  |
| B | autotune_tile_sizes_for_small_transpose_kernel | 1.046× | 1.064× | ✓ |  |
| C | increase_tile_sizes_and_fix_grid_parallelism | 1.018× | 0.893× | ✓ |  |
| D | fix_tile_sizes_for_small_xnumel_and_add_autotune | — | 0.967× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_tile_siz... | Δ | increase_tile_siz... | Δ | fix_tile_sizes_fo... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2660 ns | 277820 ns | +275160 ns (+10344.4%) | 2500 ns | -160 ns (-6.0%) | 2980 ns | +320 ns (+12.0%) | 2750 ns | +90 ns (+3.4%) |
| SM Throughput (%) | 9.19 % | 41.72 % | +32.53 pp | 5.52 % | -3.67 pp | 7.22 % | -1.97 pp | 5.47 % | -3.72 pp |
| Memory Throughput (%) | 8.68 % | 48.79 % | +40.11 pp | 5.09 % | -3.59 pp | 3.57 % | -5.11 pp | 7.44 % | -1.24 pp |
| DRAM Throughput (%) | 1.17 % | 0.01 % | -1.16 pp | 1.26 % | +0.09 pp | 1.05 % | -0.12 pp | 4.45 % | +3.28 pp |
| Achieved Occupancy (%) | 21.19 % | 24.24 % | +3.05 pp | 8.84 % | -12.35 pp | 18.40 % | -2.79 pp | 10.59 % | -10.60 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 30.00 register/thread | +2 | 28.00 register/thread | +0 | 30.00 register/thread | +2 |
| Waves/SM | 0.24 | 0.24 | +0.00 | 0.12 | -0.12 | 0.24 | +0.00 | 0.12 | -0.12 |
| No Eligible (%) | 75.47 % | 58.07 % | -17.40 pp | 74.50 % | -0.97 pp | 55.22 % | -20.25 pp | 85.18 % | +9.71 pp |
| IPC Active | 1.13 inst/cycle | 1.54 inst/cycle | +0.41 | 0.75 inst/cycle | -0.38 | 1.48 inst/cycle | +0.35 | 0.62 inst/cycle | -0.51 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.010×)

Adding a persistent while-loop with CTA-stride tile scheduling to a tiny kernel (36K elements, 2.66 µs) caused a 104x slowdown. The executed instruction count exploded from 204K to 85M because the while-loop condition check, tile index arithmetic (div/mod), and per-iteration mask recomputation are all executed inside the kernel body for every tile iteration, creating enormous overhead that dwarfs the actual memory work. The original kernel was fast precisely because it was tiny and latency-dominated by launch overhead, not by SM underutilization — the persistent loop added far more compute than it saved in wave-tail waste.

> **Applicable when**: Never applicable for tiny kernels with total work < ~1M elements and duration < ~10 µs. Persistent tile scheduling is only beneficial when: (1) total work is large enough that each CTA processes many tiles (hundreds+), (2) the kernel is memory-bandwidth bound with large working sets, and (3) the per-tile loop overhead (index arithmetic, mask computation) is negligible relative to the actual memory/compute work per tile.

> **Avoid when**: Kernel duration is under ~10 µs; total element count is under ~1M; each CTA would process only 1-2 tiles in the persistent loop (making the loop overhead dominate); the kernel is already latency-dominated by launch overhead rather than SM underutilization from insufficient waves; tl.cdiv is called at runtime inside the kernel (not as constexpr) since this generates expensive integer division instructions per warp.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_to_use_meta_tile_sizes | — | 1.000× | ✓ |  |
| B | fix_hardcoded_grid_to_use_autotuned_tile_sizes | — | 1.081× | ✓ |  |
| C | fix_autotune_grid_lambda_and_coalesce_stores | — | 0.480× | ✓ |  |
| D | revert_persistent_loop_to_2d_grid_with_autotune | — | 1.051× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_hardcoded_gri... | Δ | fix_autotune_grid... | Δ | revert_persistent... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2660 ns | 2660 ns | +0 ns (+0.0%) | 2460 ns | -200 ns (-7.5%) | 5540 ns | +2880 ns (+108.3%) | 2530 ns | -130 ns (-4.9%) |
| SM Throughput (%) | 9.19 % | 6.14 % | -3.05 pp | 6.12 % | -3.07 pp | 15.28 % | +6.09 pp | 5.65 % | -3.54 pp |
| Memory Throughput (%) | 8.68 % | 5.67 % | -3.01 pp | 5.64 % | -3.04 pp | 8.30 % | -0.38 pp | 5.21 % | -3.47 pp |
| DRAM Throughput (%) | 1.17 % | 1.18 % | +0.01 pp | 1.27 % | +0.10 pp | 0.56 % | -0.61 pp | 1.25 % | +0.08 pp |
| Achieved Occupancy (%) | 21.19 % | 9.00 % | -12.19 pp | 8.71 % | -12.48 pp | 37.45 % | +16.26 pp | 9.03 % | -12.16 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 | 30.00 register/thread | +2 | 32.00 register/thread | +4 | 30.00 register/thread | +2 |
| Waves/SM | 0.24 | 0.12 | -0.12 | 0.12 | -0.12 | 0.48 | +0.24 | 0.12 | -0.12 |
| No Eligible (%) | 75.47 % | 74.52 % | -0.95 pp | 74.94 % | -0.53 pp | 29.86 % | -45.61 pp | 75.93 % | +0.46 pp |
| IPC Active | 1.13 inst/cycle | 0.66 inst/cycle | -0.47 | 0.79 inst/cycle | -0.34 | 2.23 inst/cycle | +1.10 | 0.72 inst/cycle | -0.41 |


### Key Learnings

#### ✅ Plan D: `revert_persistent_loop_to_2d_grid_with_autotune` (1.051×)

The actual code change between before and after was minimal: adding num_stages=1 to all autotune configs and adding two new XBLOCK=1 configs (YBLOCK=64 and YBLOCK=128). The profiling metrics are nearly identical before and after (same instruction count 119040, same grid/block size 64/512, same waves_per_sm 0.12), indicating the autotuner selected the same or equivalent config in both cases. The 5.14% speedup is likely attributable to the XBLOCK=1 configs enabling better thread utilization for the narrow xnumel=9 dimension, reducing masked-thread waste, though the NCU snapshot does not capture this directly since it profiles a single kernel invocation.

> **Applicable when**: Kernel has a narrow dimension (xnumel << XBLOCK) causing high thread masking waste; autotune search space does not include XBLOCK=1 or num_stages=1; the kernel is a pointwise/elementwise op on a 2D tile grid with small x-extent and large y-extent.

> **Avoid when**: The x-dimension is large enough that XBLOCK=1 would create too many CTAs and increase launch overhead beyond the savings from reduced masking; the kernel is already bandwidth-saturated and autotune config changes have no effect; num_stages > 1 is beneficial for hiding memory latency in kernels with multiple loads.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesce_store_by_swapping_xy_tile_assignment | — | 1.157× | ✓ |  |
| B | coalesce_store_eviction_policy_and_fix_grid_lambda | — | 1.000× | ✓ |  |
| C | coalesce_stores_via_eviction_policy_and_tile_reorientation | — | 0.978× | ✓ |  |
| D | coalesce_store_by_fixing_xblock_to_xnumel_and_grid_lambda | — | 0.513× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesce_store_by... | Δ | coalesce_store_ev... | Δ | coalesce_stores_v... | Δ | coalesce_store_by... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2660 ns | 2300 ns | -360 ns (-13.5%) | 2660 ns | +0 ns (+0.0%) | 2720 ns | +60 ns (+2.3%) | 5180 ns | +2520 ns (+94.7%) |
| SM Throughput (%) | 9.19 % | 1.72 % | -7.47 pp | 6.11 % | -3.08 pp | 6.13 % | -3.06 pp | 15.77 % | +6.58 pp |
| Memory Throughput (%) | 8.68 % | 1.34 % | -7.34 pp | 5.63 % | -3.05 pp | 5.46 % | -3.22 pp | 8.57 % | -0.11 pp |
| DRAM Throughput (%) | 1.17 % | 1.34 % | +0.17 pp | 1.18 % | +0.01 pp | 2.46 % | +1.29 pp | 0.60 % | -0.57 pp |
| Achieved Occupancy (%) | 21.19 % | 8.02 % | -13.17 pp | 8.29 % | -12.90 pp | 8.45 % | -12.74 pp | 35.43 % | +14.24 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 30.00 register/thread | +2 | 30.00 register/thread | +2 | 32.00 register/thread | +4 |
| Waves/SM | 0.24 | 0.12 | -0.12 | 0.12 | -0.12 | 0.12 | -0.12 | 0.48 | +0.24 |
| No Eligible (%) | 75.47 % | 89.24 % | +13.77 pp | 79.82 % | +4.35 pp | 75.80 % | +0.33 pp | 30.01 % | -45.46 pp |
| IPC Active | 1.13 inst/cycle | 0.28 inst/cycle | -0.85 | 0.76 inst/cycle | -0.37 | 0.75 inst/cycle | -0.38 | 2.61 inst/cycle | +1.48 |

---

## Cumulative Progress

```
Baseline:  2660 ns  (1.000×)

Final best: 2750 ns  (0.9673×)
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
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},   num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},   num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 512},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 1024}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},   num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},   num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256},  num_warps=8, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4096
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +35,9 @@

     y3 = yindex
     y0 = (yindex % 64)
     y1 = yindex // 64
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
-
-
+    mask2d = ymask & xmask
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), mask=mask2d, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, mask=mask2d)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +45,7 @@

     ynumel = 4096
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3[grid](arg_0, arg_1, 4096, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3[grid](arg_0, arg_1, 4096, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
