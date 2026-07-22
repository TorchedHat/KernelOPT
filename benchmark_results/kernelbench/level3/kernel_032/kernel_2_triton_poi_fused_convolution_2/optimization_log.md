# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2780 ns — **Best latency**: 2750 ns (**1.0109×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`persistent_tile_scheduling_via_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_032/kernel_2_triton_poi_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (160,1,1), Waves/SM: 0.08
- **Registers/thread**: 28
- **Duration**: 2780 ns
- **SM Throughput**: 2.93 % | **Memory**: 6.81 % | **DRAM**: 2.47 %
- **Achieved Occupancy**: 7.53 %
- **IPC Active**: 0.29 inst/cycle | **No Eligible**: 91.14 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_via_autotune | 1.071× | 1.011× | ✓ | **✓ best** |
| B | autotune_tile_sizes_for_occupancy | 1.063× | 0.955× | ✓ |  |
| C | autotune_tile_sizes_for_underutilized_pointwise_kernel | 1.031× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2780 ns | 2750 ns | -30 ns (-1.1%) | 2910 ns | +130 ns (+4.7%) | 2780 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 2.93 % | 1.13 % | -1.80 pp | 0.53 % | -2.40 pp | 1.75 % | -1.18 pp |
| Memory Throughput (%) | 6.81 % | 1.52 % | -5.29 pp | 4.64 % | -2.17 pp | 1.56 % | -5.25 pp |
| DRAM Throughput (%) | 2.47 % | 0.34 % | -2.13 pp | 0.32 % | -2.15 pp | 0.64 % | -1.83 pp |
| Achieved Occupancy (%) | 7.53 % | 3.54 % | -3.99 pp | 1.89 % | -5.64 pp | 3.62 % | -3.91 pp |
| Registers/Thread | 28.00 register/thread | 18.00 register/thread | -10 | 17.00 register/thread | -11 | 31.00 register/thread | +3 |
| Waves/SM | 0.08 | 0.04 | -0.04 | 0.04 | -0.04 | 0.04 | -0.04 |
| No Eligible (%) | 91.14 % | 90.08 % | -1.06 pp | 93.84 % | +2.70 pp | 84.88 % | -6.26 pp |
| IPC Active | 0.29 inst/cycle | 0.15 inst/cycle | -0.14 | 0.05 inst/cycle | -0.24 | 0.35 inst/cycle | +0.06 |

---

## Iteration 2

**Best so far**: 1.0109× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_multiblock_pointwise_kernel | 0.058× | 0.025× | ✓ |  |
| B | fix_hardcoded_grid_mismatch_and_expand_autotune_configs | 1.042× | 1.045× | ✓ |  |
| C | fix_uncoalesced_access_via_xblock_yblock_swap_and_larger_tiles | 1.057× | 0.955× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_multib... | Δ | fix_hardcoded_gri... | Δ | fix_uncoalesced_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2780 ns | 111460 ns | +108680 ns (+3909.4%) | 2660 ns | -120 ns (-4.3%) | 2910 ns | +130 ns (+4.7%) |
| SM Throughput (%) | 2.93 % | 12.48 % | +9.55 pp | 1.71 % | -1.22 pp | 3.02 % | +0.09 pp |
| Memory Throughput (%) | 6.81 % | 19.33 % | +12.52 pp | 1.52 % | -5.29 pp | 6.96 % | +0.15 pp |
| DRAM Throughput (%) | 2.47 % | 0.06 % | -2.41 pp | 0.67 % | -1.80 pp | 2.37 % | -0.10 pp |
| Achieved Occupancy (%) | 7.53 % | 7.59 % | +0.06 pp | 3.66 % | -3.87 pp | 7.05 % | -0.48 pp |
| Registers/Thread | 28.00 register/thread | 31.00 register/thread | +3 | 31.00 register/thread | +3 | 34.00 register/thread | +6 |
| Waves/SM | 0.08 | 0.08 | +0.00 | 0.04 | -0.04 | 0.10 | +0.02 |
| No Eligible (%) | 91.14 % | 87.07 % | -4.07 pp | 84.21 % | -6.93 pp | 88.89 % | -2.25 pp |
| IPC Active | 0.29 inst/cycle | 0.52 inst/cycle | +0.23 | 0.27 inst/cycle | -0.02 | 0.32 inst/cycle | +0.03 |


### Key Learnings

#### ❌ Plan A: `persistent_multiblock_pointwise_kernel` (0.025×)

Adding a persistent while-loop inside a Triton kernel that iterates over tiles caused a 40x slowdown despite the original kernel being launch-latency bound (0.04 waves/SM, 3.54% occupancy). The loop body executed 11.1M instructions vs 21.9K before — a 500x increase — because the grid was not reduced to match the persistent pattern: the launcher still used the original 2D grid (160 CTAs) with XBLOCK=128/YBLOCK=8, so each CTA iterated over all 40 tiles instead of a single tile, multiplying total work by ~40x. The implementation flaw was that the grid size was not changed to a small persistent grid (e.g., 40 or 528 CTAs with loop stride = grid_size), so the persistent loop ran redundantly across all original CTAs rather than distributing work.

> **Applicable when**: Never apply this pattern as implemented. The correct persistent kernel pattern requires reducing the grid to fewer CTAs than total_tiles so each CTA processes multiple tiles.

> **Avoid when**: Do not add a persistent while-loop to a kernel without simultaneously reducing the grid size to a small persistent count (e.g., num_SMs * blocks_per_SM). If the grid size equals or exceeds total_tiles, the loop body executes at most once per CTA and the overhead of the loop + increased register pressure + shared memory allocation causes severe regression. Especially harmful for tiny kernels (< 100us, < 10K elements) where launch overhead is already the dominant cost.

---

## Iteration 3

**Best so far**: 1.0109× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_with_lambda_and_large_tile_configs | 1.055× | 1.112× | ✓ |  |
| B | fix_hardcoded_grid_and_kernel_constant_mismatch_with_lambda | 1.056× | 0.986× | ✓ |  |
| C | revert_broken_persistent_loop_restore_single_tile_with_autotune | 1.008× | 1.045× | ✓ |  |
| D | add_triton_autotune | 1.028× | 0.986× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ | revert_broken_per... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2780 ns | 2500 ns | -280 ns (-10.1%) | 2820 ns | +40 ns (+1.4%) | 2660 ns | -120 ns (-4.3%) | 2820 ns | +40 ns (+1.4%) |
| SM Throughput (%) | 2.93 % | 1.75 % | -1.18 pp | 2.86 % | -0.07 pp | 3.28 % | +0.35 pp | 3.59 % | +0.66 pp |
| Memory Throughput (%) | 6.81 % | 1.58 % | -5.23 pp | 7.22 % | +0.41 pp | 4.72 % | -2.09 pp | 4.46 % | -2.35 pp |
| DRAM Throughput (%) | 2.47 % | 0.72 % | -1.75 pp | 2.42 % | -0.05 pp | 2.60 % | +0.13 pp | 2.42 % | -0.05 pp |
| Achieved Occupancy (%) | 7.53 % | 3.62 % | -3.91 pp | 6.89 % | -0.64 pp | 7.14 % | -0.39 pp | 7.03 % | -0.50 pp |
| Registers/Thread | 28.00 register/thread | 31.00 register/thread | +3 | 34.00 register/thread | +6 | 30.00 register/thread | +2 | 32.00 register/thread | +4 |
| Waves/SM | 0.08 | 0.04 | -0.04 | 0.10 | +0.02 | 0.08 | +0.00 | 0.08 | +0.00 |
| No Eligible (%) | 91.14 % | 84.60 % | -6.54 pp | 88.46 % | -2.68 pp | 86.91 % | -4.23 pp | 88.52 % | -2.62 pp |
| IPC Active | 0.29 inst/cycle | 0.34 inst/cycle | +0.05 | 0.42 inst/cycle | +0.13 | 0.39 inst/cycle | +0.10 | 0.49 inst/cycle | +0.20 |

---

## Cumulative Progress

```
Baseline:  2780 ns  (1.000×)
Iter 1     2750 ns  (1.011×)  — persistent_tile_scheduling_via_autotune

Final best: 2750 ns  (1.0109×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,23 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 64},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 16},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 32},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 64},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 256}, num_warps=8, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 1280
@@ -20,11 +37,9 @@

     y1 = yindex // 128
     y3 = yindex
     tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 8192*y1), xmask & ymask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr1 + (y0), xmask & ymask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tl.store(out_ptr0 + (x2 + 64*y3), tmp2, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -33,7 +48,7 @@

     ynumel = 1280
     xnumel = 64
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 1280, 64, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 1280, 64)
     return arg_2
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
