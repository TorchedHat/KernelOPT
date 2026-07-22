# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_36` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2850 ns — **Best latency**: 2340 ns (**1.2179×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_small_grid_utilization`** beats baseline latency.

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
- **Duration**: 2850 ns
- **SM Throughput**: 1.83 % | **Memory**: 1.75 % | **DRAM**: 0.59 %
- **Achieved Occupancy**: 6.11 %
- **IPC Active**: 0.36 inst/cycle | **No Eligible**: 90.06 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_occupancy | 1.071× | 1.188× | ✓ |  |
| B | autotune_tile_sizes_for_small_grid_utilization | 1.073× | 1.218× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2850 ns | 2400 ns | -450 ns (-15.8%) | 2340 ns | -510 ns (-17.9%) |
| SM Throughput (%) | 1.83 % | 0.17 % | -1.66 pp | 0.33 % | -1.50 pp |
| Memory Throughput (%) | 1.75 % | 1.56 % | -0.19 pp | 0.64 % | -1.11 pp |
| DRAM Throughput (%) | 0.59 % | 0.11 % | -0.48 pp | 0.47 % | -0.12 pp |
| Achieved Occupancy (%) | 6.11 % | 1.56 % | -4.55 pp | 3.03 % | -3.08 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.05 | 0.02 | -0.03 | 0.02 | -0.03 |
| No Eligible (%) | 90.06 % | 95.78 % | +5.72 pp | 92.17 % | +2.11 pp |
| IPC Active | 0.36 inst/cycle | 0.03 inst/cycle | -0.33 | 0.13 inst/cycle | -0.23 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_for_small_grid_utilization` (1.218×)

When a kernel's problem dimension (xnumel) is much smaller than the fixed XBLOCK tile size, the majority of thread lanes are wasted on out-of-bounds masked work. Adding a @triton.autotune decorator with configs that sweep XBLOCK down to match the small dimension (e.g., XBLOCK=1,4,8 when xnumel=25) dramatically reduces wasted instructions and elapsed cycles. The speedup here (1.22x) came primarily from a 7x reduction in executed instructions (38784 → 5448) and a corresponding drop in kernel duration (2.85 µs → 2.34 µs), even though grid coverage and achieved occupancy did not improve as predicted.

> **Applicable when**: A Triton kernel has a fixed XBLOCK (or YBLOCK) that is much larger than the actual problem dimension (xnumel or ynumel), causing the majority of thread lanes to be masked off. Signals: high executed instruction count relative to problem size, low achieved occupancy, LaunchConfiguration and AchievedOccupancy NCU rules triggered, xnumel << XBLOCK (e.g., xnumel=25 with XBLOCK=128).

> **Avoid when**: The problem dimensions are large enough to fill the tile (xnumel ≥ XBLOCK), or when the grid is already large enough to saturate all SMs (waves_per_sm ≥ 1). Also avoid when the kernel body has complex shared-memory usage that depends on a specific XBLOCK/YBLOCK value.

---

## Cumulative Progress

```
Baseline:  2850 ns  (1.000×)
Iter 1     2340 ns  (1.218×)  — autotune_tile_sizes_for_small_grid_utilization

Final best: 2340 ns  (1.2179×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 16},  num_warps=2),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 16},  num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 8},   num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 16},  num_warps=4),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 8},   num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_36(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 768
@@ -22,15 +38,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 16*x2 + 400*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 768
     xnumel = 25
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_36[grid](arg_0, arg_1, 768, 25, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_36[grid](arg_0, arg_1, 768, 25)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
