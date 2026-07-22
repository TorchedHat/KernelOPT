# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_45` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4640 ns — **Best latency**: 3840 ns (**1.2083×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_eliminate_partial_wave`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_45`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_45_triton_poi_fused_convolution_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3136,1,1), Waves/SM: 1.48
- **Registers/thread**: 28
- **Duration**: 4640 ns
- **SM Throughput**: 35.74 % | **Memory**: 31.52 % | **DRAM**: 4.01 %
- **Achieved Occupancy**: 71.30 %
- **IPC Active**: 2.55 inst/cycle | **No Eligible**: 30.31 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_tile_sizes_to_eliminate_partial_wave_and_thread_waste | 0.990× | 1.074× | ✓ |  |
| B | autotune_tile_sizes_eliminate_partial_wave | 1.043× | 1.208× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_tile_sizes_to... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4640 ns | 4320 ns | -320 ns (-6.9%) | 3840 ns | -800 ns (-17.2%) |
| SM Throughput (%) | 35.74 % | 15.34 % | -20.40 pp | 6.60 % | -29.14 pp |
| Memory Throughput (%) | 31.52 % | 11.36 % | -20.16 pp | 4.83 % | -26.69 pp |
| DRAM Throughput (%) | 4.01 % | 4.29 % | +0.28 pp | 4.83 % | +0.82 pp |
| Achieved Occupancy (%) | 71.30 % | 47.64 % | -23.66 pp | 26.62 % | -44.68 pp |
| Registers/Thread | 28.00 register/thread | 22.00 register/thread | -6 | 16.00 register/thread | -12 |
| Waves/SM | 1.48 | 0.74 | -0.74 | 0.74 | -0.74 |
| No Eligible (%) | 30.31 % | 41.59 % | +11.28 pp | 69.21 % | +38.90 pp |
| IPC Active | 2.55 inst/cycle | 2.04 inst/cycle | -0.51 | 1.04 inst/cycle | -1.51 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_eliminate_partial_wave` (1.208×)

When a Triton kernel's XBLOCK is much larger than xnumel (e.g., XBLOCK=128 for xnumel=9), the vast majority of X-dimension threads are idle, inflating the grid and creating a large partial wave. Adding @triton.autotune with configs that sweep XBLOCK down to near xnumel and increase YBLOCK reduces total block count, cuts the partial-wave tail overhead flagged by the LaunchConfiguration NCU rule, and improves L1 hit rate (28% → 47%) by processing larger contiguous Y tiles per block. The speedup (1.21x, duration 4.64→3.84 µs) comes primarily from reducing instruction count (~7x fewer executed instructions) and eliminating wasted thread slots, though achieved occupancy also dropped (71%→27%), indicating the kernel is now too small to fully saturate the GPU.

> **Applicable when**: LaunchConfiguration NCU rule fires with estimated speedup ≥ 20%; XBLOCK >> xnumel (e.g., ratio > 5x); kernel has a partial wave accounting for ≥ 30% of runtime; waves_per_sm between 1.0 and 2.0; achieved occupancy < theoretical occupancy by > 20pp.

> **Avoid when**: xnumel is already close to or larger than XBLOCK (tile is well-matched); kernel is already occupancy-limited from the other direction (too few warps); grid is already small (< 1 wave) — reducing block size further would worsen wave coverage; kernel is compute-bound rather than launch-configuration-bound.

---

## Cumulative Progress

```
Baseline:  4640 ns  (1.000×)
Iter 1     3840 ns  (1.208×)  — autotune_tile_sizes_eliminate_partial_wave

Final best: 3840 ns  (1.2083×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 2,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 2,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_45(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 25088
@@ -22,15 +40,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 112*x2 + 1008*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 25088
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_45[grid](arg_0, arg_1, 25088, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_45[grid](arg_0, arg_1, 25088, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
