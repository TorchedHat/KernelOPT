# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_57` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5920 ns — **Best latency**: 5150 ns (**1.1495×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`fix_tile_dimensions_for_coalesced_vectorized_access`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_57`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_57_triton_poi_fused_convolution_5/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (5184,1,1), Waves/SM: 2.45
- **Registers/thread**: 28
- **Duration**: 5920 ns
- **SM Throughput**: 45.91 % | **Memory**: 41.64 % | **DRAM**: 5.17 %
- **Achieved Occupancy**: 72.49 %
- **IPC Active**: 2.82 inst/cycle | **No Eligible**: 26.51 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_partial_wave_via_autotune_tile_sizes | 1.051× | 1.109× | ✓ |  |
| C | fix_tile_sizes_and_add_autotune_for_transpose_kernel | 1.054× | 1.134× | ✓ |  |
| D | fix_tile_dimensions_for_coalesced_vectorized_access | 1.087× | 1.149× | ✓ | **✓ best** |
| E | add_triton_autotune | 1.018× | 1.010× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_partial... | Δ | fix_tile_sizes_an... | Δ | fix_tile_dimensio... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5920 ns | 5340 ns | -580 ns (-9.8%) | 5220 ns | -700 ns (-11.8%) | 5150 ns | -770 ns (-13.0%) | 5860 ns | -60 ns (-1.0%) |
| SM Throughput (%) | 45.91 % | 13.92 % | -31.99 pp | 28.52 % | -17.39 pp | 14.64 % | -31.27 pp | 47.29 % | +1.38 pp |
| Memory Throughput (%) | 41.64 % | 5.73 % | -35.91 pp | 8.48 % | -33.16 pp | 5.93 % | -35.71 pp | 42.93 % | +1.29 pp |
| DRAM Throughput (%) | 5.17 % | 5.73 % | +0.56 pp | 5.89 % | +0.72 pp | 5.93 % | +0.76 pp | 5.26 % | +0.09 pp |
| Achieved Occupancy (%) | 72.49 % | 25.20 % | -47.29 pp | 29.72 % | -42.77 pp | 26.08 % | -46.41 pp | 71.87 % | -0.62 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 28.00 register/thread | +0 |
| Waves/SM | 2.45 | 2.45 | +0.00 | 4.91 | +2.46 | 2.45 | +0.00 | 2.45 | +0.00 |
| No Eligible (%) | 26.51 % | 62.62 % | +36.11 pp | 47.76 % | +21.25 pp | 62.53 % | +36.02 pp | 26.23 % | -0.28 pp |
| IPC Active | 2.82 inst/cycle | 1.30 inst/cycle | -1.52 | 1.96 inst/cycle | -0.86 | 1.28 inst/cycle | -1.54 | 2.96 inst/cycle | +0.14 |

---

## Iteration 2

**Best so far**: 1.1495× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.003× | 0.002× | ✓ |  |
| D | persistent_tile_scheduling | 0.001× | 0.000× | ✓ |  |
| E | add_triton_autotune | 0.477× | 0.379× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5920 ns | 2830000 ns | +2824080 ns (+47704.1%) | 16920000 ns | +16914080 ns (+285710.8%) | 15620 ns | +9700 ns (+163.9%) |
| SM Throughput (%) | 45.91 % | 62.07 % | +16.16 pp | 68.92 % | +23.01 pp | 61.41 % | +15.50 pp |
| Memory Throughput (%) | 41.64 % | 82.29 % | +40.65 pp | 90.29 % | +48.65 pp | 74.43 % | +32.79 pp |
| DRAM Throughput (%) | 5.17 % | 0.01 % | -5.16 pp | 0.01 % | -5.16 pp | 1.96 % | -3.21 pp |
| Achieved Occupancy (%) | 72.49 % | 47.22 % | -25.27 pp | 87.37 % | +14.88 pp | 54.80 % | -17.69 pp |
| Registers/Thread | 28.00 register/thread | 64.00 register/thread | +36 | 32.00 register/thread | +4 | 48.00 register/thread | +20 |
| Waves/SM | 2.45 | 4.91 | +2.46 | 2.45 | +0.00 | 3.93 | +1.48 |
| No Eligible (%) | 26.51 % | 38.54 % | +12.03 pp | 29.87 % | +3.36 pp | 28.33 % | +1.82 pp |
| IPC Active | 2.82 inst/cycle | 2.46 inst/cycle | -0.36 | 2.81 inst/cycle | -0.01 | 2.85 inst/cycle | +0.03 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.002×)

Converting a 2D grid Triton kernel to a persistent 1D grid with an intra-kernel while-loop caused a catastrophic regression: executed instructions exploded from 584K to 1.37 billion (~2340x), and duration increased from 5.15 µs to 2.83 ms (~550x slower). The while-loop inside the Triton JIT kernel forces the compiler to serialize all tile iterations within a single CTA rather than parallelizing them across the GPU, eliminating the GPU's ability to overlap tile execution across SMs. Additionally, registers per thread jumped from 16 to 64 (4x), halving theoretical occupancy from 100% to 50% and triggering the TheoreticalOccupancy NCU rule. The persistent pattern is fundamentally incompatible with Triton's execution model for this workload: the original 5184-CTA grid already allows the GPU scheduler to overlap CTA execution efficiently, and collapsing to 132 CTAs with a serial loop removes that parallelism entirely.

> **Applicable when**: Never: this strategy caused a ~550x regression. Do not apply persistent while-loop scheduling to short-duration Triton pointwise/transpose kernels.

> **Avoid when**: Always avoid for short-duration kernels (< 100 µs). The persistent while-loop pattern in Triton serializes tile iterations within each CTA, causing instruction count to multiply by the number of loop iterations. This eliminates GPU-level parallelism across tiles. The pattern also inflates register usage (16→64 registers/thread) which halves theoretical occupancy. Only consider persistent kernels in CUDA (not Triton) where the loop body can be overlapped with memory prefetching via explicit pipeline stages.

#### ❌ Plan D: `persistent_tile_scheduling` (0.002×)

Converting a 2D grid Triton kernel to a persistent 1D grid with an intra-kernel while-loop caused a catastrophic regression: executed instructions exploded from 584K to 1.37 billion (~2340x), and duration increased from 5.15 µs to 2.83 ms (~550x slower). The while-loop inside the Triton JIT kernel forces the compiler to serialize all tile iterations within a single CTA rather than parallelizing them across the GPU, eliminating the GPU's ability to overlap tile execution across SMs. Additionally, registers per thread jumped from 16 to 64 (4x), halving theoretical occupancy from 100% to 50% and triggering the TheoreticalOccupancy NCU rule. The persistent pattern is fundamentally incompatible with Triton's execution model for this workload: the original 5184-CTA grid already allows the GPU scheduler to overlap CTA execution efficiently, and collapsing to 132 CTAs with a serial loop removes that parallelism entirely.

> **Applicable when**: Never: this strategy caused a ~550x regression. Do not apply persistent while-loop scheduling to short-duration Triton pointwise/transpose kernels.

> **Avoid when**: Always avoid for short-duration kernels (< 100 µs). The persistent while-loop pattern in Triton serializes tile iterations within each CTA, causing instruction count to multiply by the number of loop iterations. This eliminates GPU-level parallelism across tiles. The pattern also inflates register usage (16→64 registers/thread) which halves theoretical occupancy. Only consider persistent kernels in CUDA (not Triton) where the loop body can be overlapped with memory prefetching via explicit pipeline stages.

---

## Iteration 3

**Best so far**: 1.1495× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | remove_persistent_while_loop_restore_standard_2d_grid_with_register_cap | 1.030× | 0.603× | ✓ |  |
| D | add_triton_autotune | 0.003× | 0.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | remove_persistent... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5920 ns | 9820 ns | +3900 ns (+65.9%) | 2860000 ns | +2854080 ns (+48210.8%) |
| SM Throughput (%) | 45.91 % | 65.49 % | +19.58 pp | 61.74 % | +15.83 pp |
| Memory Throughput (%) | 41.64 % | 55.20 % | +13.56 pp | 81.81 % | +40.17 pp |
| DRAM Throughput (%) | 5.17 % | 3.11 % | -2.06 pp | 0.02 % | -5.15 pp |
| Achieved Occupancy (%) | 72.49 % | 79.42 % | +6.93 pp | 47.24 % | -25.25 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 64.00 register/thread | +36 |
| Waves/SM | 2.45 | 2.45 | +0.00 | 4.91 | +2.46 |
| No Eligible (%) | 26.51 % | 17.97 % | -8.54 pp | 38.52 % | +12.01 pp |
| IPC Active | 2.82 inst/cycle | 3.26 inst/cycle | +0.44 | 2.45 inst/cycle | -0.37 |

---

## Cumulative Progress

```
Baseline:  5920 ns  (1.000×)
Iter 1     5150 ns  (1.149×)  — fix_tile_dimensions_for_coalesced_vectorized_access

Final best: 5150 ns  (1.1495×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'YBLOCK': 32,  'XBLOCK': 128}, num_warps=4),
+        triton.Config({'YBLOCK': 64,  'XBLOCK': 256}, num_warps=4),
+        triton.Config({'YBLOCK': 64,  'XBLOCK': 512}, num_warps=8),
+        triton.Config({'YBLOCK': 128, 'XBLOCK': 1024}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_57(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 41472
@@ -22,15 +32,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 144*x2 + 1296*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 41472
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_57[grid](arg_0, arg_1, 41472, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_57[grid](arg_0, arg_1, 41472, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
